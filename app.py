from pathlib import Path
import hashlib
from tempfile import TemporaryDirectory
import streamlit as st
from ensemble import Ensemble
from run_status import running
from solvent_rules import display_solvent

st.set_page_config(page_title="TLC Studio", page_icon="⚗", layout="centered")
st.markdown("""
<style>
.stApp {background:#f8f6ef;color:#152921;font-family:Arial, sans-serif}
.block-container {max-width:54rem;padding-top:3rem}
h1 {font-family:Arial,sans-serif;font-weight:800;letter-spacing:-.06em}
.intro {font:italic 1.17rem/1.5 Georgia,serif;color:#171717}
textarea, [data-testid="stCheckbox"] label p, [data-testid="stNumberInput"] input {font-size:1.08rem!important}
[data-testid="stSidebar"] [data-testid="stRadio"] > label p {font-size:1.45rem;font-weight:700}
[data-testid="stSidebar"] [role="radiogroup"] label {min-height:52px;padding:.35rem .6rem;align-items:center;width:100%;border-radius:.6rem}
[data-testid="stSidebar"] [role="radiogroup"] label p {font-size:1.28rem;font-weight:550;line-height:1.3}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#dce7dd}
.run-status {display:flex;gap:.65rem;align-items:center;font-weight:600}
.run-status span {font-size:1.7rem;display:inline-block;transform-origin:center bottom;animation:tlc-flame .65s ease-in-out infinite alternate}
@keyframes tlc-flame {from {transform:rotate(-8deg) scale(.91)} to {transform:rotate(6deg) scale(1.09)}}
@media (prefers-reduced-motion:reduce) {.run-status span {animation:none}}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models = Path(__file__).parent / "models"
    mpnn = models / "MPNN.pt"
    mpnn_hash = hashlib.sha256(mpnn.read_bytes()).hexdigest()
    if mpnn_hash != "54d46ae50ec5027e2a07f9dd43cbfeb4481a9dc5f9436f104cd7f924d2dca4be":
        raise ValueError("MPNN checkpoint checksum mismatch")
    with TemporaryDirectory() as tmp:
        attentive = Path(tmp) / "AttentiveFP.pt"
        digest = hashlib.sha256()
        with attentive.open("wb") as output:
            for index in range(5):
                part = (models / f"AttentiveFP.pt.part{index}").read_bytes()
                digest.update(part)
                output.write(part)
        if digest.hexdigest() != "b0ce3c93834d03578d094f7f7b7dc2429fb35892d6dd4d36478c8f27f909d5ba":
            raise ValueError("AttentiveFP checkpoint checksum mismatch")
        ensemble = Ensemble(attentive, mpnn)
        ensemble.checkpoint_hashes = {
            "AttentiveFP": digest.hexdigest(),
            "MPNN": mpnn_hash,
        }
        return ensemble

predictor = load_models()
page = st.sidebar.radio("TLC Studio", ["Predict Rf", "Mobile Phase Suggestions", "Find molecules"])
if page == "Find molecules":
    from molecule_search_page import render_molecule_search
    render_molecule_search(predictor)
    st.stop()
if page == "Mobile Phase Suggestions":
    from search_page import render_search
    render_search(predictor, natural_products=True)
    st.stop()

st.title("TLC Studio")
st.markdown('<p class="intro">Predict the retention factor of a molecule in your solvent systems.</p>', unsafe_allow_html=True)
for smiles, names, fractions, result in st.session_state.get("history", []):
    with st.chat_message("user"):
        st.code(smiles, language=None)
        st.text(" · ".join(f"{display_solvent(s)} {v:.1%}" for s,v in zip(names, fractions)))
    with st.chat_message("assistant"):
        st.metric("Predicted Rf", f"{result['Rf']:.4f}")
        st.caption(f"AttentiveFP {result['AttentiveFP']:.4f}   ·   MPNN {result['MPNN']:.4f}")

smiles = st.text_area("Molecular SMILES", height=100, placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
st.subheader("Mobile phase")
st.caption("Tick each solvent and enter its relative parts. For a 6:1 mixture, enter 6 and 1.")
names, parts = [], []
with st.container(height=340, border=True):
    for solvent in predictor.solvents:
        choice, ratio = st.columns([3,2])
        selected = choice.checkbox(display_solvent(solvent), key=f"predict_solvent_{solvent}")
        value = ratio.number_input(f"Parts of {solvent}", min_value=0., value=1., step=0.1,
                                   disabled=not selected, label_visibility="collapsed")
        if selected:
            names.append(solvent)
            parts.append(value)
if st.button("Predict Rf", type="primary", use_container_width=True):
    try:
        with running("Predicting Rf…"):
            result = predictor.predict(smiles, names, ":".join(map(str,parts)))
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.session_state.setdefault("history", []).append(
            (result["SMILES"], names, list(result["mixture"].values()), result))
        st.rerun()

