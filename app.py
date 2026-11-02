import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date
import re

st.set_page_config(page_title="Medical Report Consistency Checker", page_icon="🩺", layout="wide")

DATA = Path(__file__).parent / "data"

@st.cache_data
def load_data():
    names = ["patients", "conditions", "allergies", "medications", "observations", "encounters"]
    return {name: pd.read_csv(DATA / f"{name}.csv", low_memory=False) for name in names}

data = load_data()
patients = data["patients"].copy()
conditions = data["conditions"].copy()
allergies = data["allergies"].copy()
medications = data["medications"].copy()
observations = data["observations"].copy()
encounters = data["encounters"].copy()

def clean_date(series):
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)

for frame, col in [(conditions,"START"),(conditions,"STOP"),(allergies,"START"),(allergies,"STOP"),
                   (medications,"START"),(medications,"STOP"),(observations,"DATE"),
                   (encounters,"START"),(encounters,"STOP")]:
    if col in frame.columns:
        frame[col+"_DT"] = clean_date(frame[col])

st.title("🩺 AI Medical Report Consistency Checker")
st.caption("Prototype v1 | Synthea synthetic patient data | Evidence-linked review flags")
st.info("This is a research prototype, not a diagnostic tool. The current detection engine is a transparent rule-based baseline; no clinical AI model has been trained or validated yet.")

with st.sidebar:
    st.header("Patient selection")
    patient_ids = sorted(patients["Id"].dropna().astype(str).unique().tolist())
    # Default to a patient with a richer record history and at least one allergy entry.
preferred_patient = "abbe7ce8-820b-cb76-8d0c-02fe9d9decb0"
default_index = patient_ids.index(preferred_patient) if preferred_patient in patient_ids else 0
selected = st.selectbox("Choose a synthetic patient", patient_ids, index=default_index,
                        format_func=lambda x: f"Patient {x[:8]}…")
    st.divider()
    st.markdown("**Dataset**")
    st.write(f"{len(patients):,} synthetic patients")
    st.write(f"{len(conditions):,} condition records")
    st.write(f"{len(allergies):,} allergy records")
    st.write(f"{len(medications):,} medication records")
    st.write(f"{len(observations):,} observations")

p = patients[patients["Id"].astype(str) == selected].iloc[0]
def subset(frame):
    return frame[frame["PATIENT"].astype(str) == selected].copy()

pc = subset(conditions)
pa = subset(allergies)
pm = subset(medications)
po = subset(observations)
pe = subset(encounters)

# Keep demographics minimal; never display direct identifiers such as SSN/passport/address.
birth = pd.to_datetime(p.get("BIRTHDATE"), errors="coerce")
age = None
if pd.notna(birth):
    age = max(0, int((pd.Timestamp.today() - birth).days / 365.25))

c1,c2,c3,c4 = st.columns(4)
c1.metric("Conditions", len(pc))
c2.metric("Allergy records", len(pa))
c3.metric("Medication records", len(pm))
c4.metric("Observations", len(po))
st.caption(f"Selected synthetic patient • {p.get('GENDER','Not recorded')} • Approx. age {age if age is not None else 'unknown'}")

tab1, tab2, tab3, tab4 = st.tabs(["Patient overview", "Timeline", "Consistency checks", "Model demonstration"])

with tab1:
    st.subheader("Clinical record summary")
    left,right = st.columns(2)
    with left:
        st.markdown("#### Conditions")
        if len(pc):
            view = pc[["START","STOP","DESCRIPTION"]].sort_values("START")
            st.dataframe(view, use_container_width=True, hide_index=True)
        else:
            st.write("No condition records for this patient.")
        st.markdown("#### Allergies")
        if len(pa):
            cols = [c for c in ["START","STOP","DESCRIPTION","CATEGORY","REACTION1","SEVERITY1"] if c in pa.columns]
            st.dataframe(pa[cols].sort_values("START"), use_container_width=True, hide_index=True)
        else:
            st.write("No allergy records for this patient.")
    with right:
        st.markdown("#### Medications")
        if len(pm):
            cols = [c for c in ["START","STOP","DESCRIPTION","REASONDESCRIPTION"] if c in pm.columns]
            st.dataframe(pm[cols].sort_values("START"), use_container_width=True, hide_index=True)
        else:
            st.write("No medication records for this patient.")
        st.markdown("#### Recent observations")
        if len(po):
            cols = [c for c in ["DATE","DESCRIPTION","VALUE","UNITS"] if c in po.columns]
            st.dataframe(po[cols].sort_values("DATE", ascending=False).head(25), use_container_width=True, hide_index=True)
        else:
            st.write("No observation records for this patient.")

with tab2:
    st.subheader("Longitudinal patient timeline")
    events = []
    for _,r in pc.iterrows():
        events.append({"Date":r.get("START"),"Type":"Condition","Event":r.get("DESCRIPTION")})
    for _,r in pa.iterrows():
        events.append({"Date":r.get("START"),"Type":"Allergy","Event":r.get("DESCRIPTION")})
    for _,r in pm.iterrows():
        events.append({"Date":r.get("START"),"Type":"Medication","Event":r.get("DESCRIPTION")})
    for _,r in po.iterrows():
        events.append({"Date":r.get("DATE"),"Type":"Observation","Event":f"{r.get('DESCRIPTION')}: {r.get('VALUE')} {r.get('UNITS','')}"})
    timeline = pd.DataFrame(events)
    if len(timeline):
        timeline["Date"] = pd.to_datetime(timeline["Date"], errors="coerce", utc=True)
        timeline = timeline.dropna(subset=["Date"]).sort_values("Date")
        st.dataframe(timeline.tail(100), use_container_width=True, hide_index=True)
        st.caption("Showing up to the latest 100 dated events for readability.")
    else:
        st.warning("No dated events available for this patient.")

def run_checks():
    findings = []
    # Rule 1: Same patient, same observation, same calendar day, different numeric values.
    if len(po):
        temp = po.copy()
        temp["DAY"] = temp["DATE_DT"].dt.date
        temp["NUM_VALUE"] = pd.to_numeric(temp["VALUE"], errors="coerce")
        grouped = temp.dropna(subset=["DAY","NUM_VALUE"]).groupby(["DESCRIPTION","UNITS","DAY"])
        for (desc,unit,day), grp in grouped:
            vals = sorted(grp["NUM_VALUE"].unique().tolist())
            if len(vals) > 1:
                findings.append({
                    "severity":"Review",
                    "category":"Same-day observation disagreement",
                    "date":str(day),
                    "summary":f"{desc} has multiple values on the same day ({', '.join(map(str, vals))} {unit}).",
                    "evidence":f"{len(grp)} source observation rows; values: {vals} {unit}.",
                    "interpretation":"Potential documentation or measurement discrepancy. Check source context, specimen and encounter."
                })
    # Rule 2: Exact allergy description has both active and ended records.
    if len(pa):
        temp = pa.copy()
        temp["KEY"] = temp["DESCRIPTION"].fillna("").str.lower().str.strip()
        for key, grp in temp[temp["KEY"]!=""].groupby("KEY"):
            has_active = grp["STOP"].isna().any()
            has_ended = grp["STOP"].notna().any()
            if has_active and has_ended:
                findings.append({
                    "severity":"Review",
                    "category":"Allergy status differs across records",
                    "date":"Multiple dates",
                    "summary":f"'{grp.iloc[0]['DESCRIPTION']}' appears in both active and ended allergy records.",
                    "evidence":f"{len(grp)} allergy records; start dates: {', '.join(grp['START'].astype(str).head(4))}.",
                    "interpretation":"This may reflect a legitimate status update or inconsistent documentation; it is not proof of an error."
                })
    # Rule 3: Exact medication has overlapping active records.
    if len(pm):
        temp = pm.copy()
        temp["KEY"] = temp["DESCRIPTION"].fillna("").str.lower().str.strip()
        for key, grp in temp[temp["KEY"]!=""].groupby("KEY"):
            active = grp[grp["STOP"].isna()]
            if len(active) > 1:
                findings.append({
                    "severity":"Review",
                    "category":"Repeated active medication entries",
                    "date":"Multiple dates",
                    "summary":f"'{active.iloc[0]['DESCRIPTION']}' has {len(active)} records without a stop date.",
                    "evidence":f"Start dates: {', '.join(active['START'].astype(str).head(5))}.",
                    "interpretation":"May represent duplicate documentation, repeated prescriptions or ongoing therapy. Verify encounter and dosage context."
                })
    return findings

with tab3:
    st.subheader("Rule-based consistency analysis")
    st.write("The baseline checks for same-day measurement disagreements, allergy records with differing active/ended status, and repeated active medication entries.")
    if st.button("Run consistency checks", type="primary"):
        st.session_state["findings"] = run_checks()
    findings = st.session_state.get("findings")
    if findings is None:
        st.info("Select **Run consistency checks** to analyse this patient's records.")
    elif not findings:
        st.success("No issues were identified by the current limited rule set. This does not prove the records are fully consistent.")
    else:
        st.warning(f"{len(findings)} item(s) require review.")
        for i,f in enumerate(findings,1):
            with st.expander(f"{i}. {f['category']} — {f['severity']}", expanded=True):
                st.write(f"**Summary:** {f['summary']}")
                st.write(f"**Date:** {f['date']}")
                st.write(f"**Evidence:** {f['evidence']}")
                st.write(f"**Interpretation:** {f['interpretation']}")

with tab4:
    st.subheader("Illustrative contradiction example")
    st.caption("A controlled synthetic example to demonstrate the planned semantic/NLI stage. It is not a finding extracted from the selected Synthea patient.")
    a = "The patient has a documented allergy to penicillin."
    b = "The patient has no known allergy to penicillin."
    st.markdown("**Statement A — Earlier report**")
    st.code(a)
    st.markdown("**Statement B — Later report**")
    st.code(b)
    if st.button("Run baseline phrase check"):
        # transparent rule demonstration, not a trained NLI model
        negation = bool(re.search(r"\b(no|not|without|denies|negative)\b", b.lower()))
        same_concept = "penicillin" in a.lower() and "penicillin" in b.lower()
        if negation and same_concept:
            st.warning("Potential contradiction pattern detected.")
            st.write("The statements mention the same concept, while the later statement contains an explicit negation.")
            st.caption("This is a simple demonstration rule, not a medical NLI model prediction.")
        else:
            st.info("No contradiction pattern detected by this simple rule.")
    st.markdown("#### Planned model upgrade")
    st.write("A clinical NLI transformer will replace this demonstration rule after model selection and validation. It will score premise–hypothesis pairs as entailment, contradiction or neutral. Temporal logic and source review will still be needed.")

st.divider()
st.caption("Research note: Synthea records are synthetic. Current rules are deliberately conservative and provide review flags, not diagnoses or treatment advice.")
