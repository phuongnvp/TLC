import streamlit as st
from solvent_search import search
from run_status import running
from solvent_rules import NATURAL_SOLVENTS, natural_solvent_pool, solvent_label


def render_search(ensemble, natural_products=False):
    st.title("Mobile Phase Suggestions")
    st.markdown('<p class="intro">Search for a mobile phase that brings your molecule close to a target Rf.</p>', unsafe_allow_html=True)
    if natural_products:
        #st.subheader("Common solvents for natural products")
        available = natural_solvent_pool(ensemble.solvents)
        #st.write(" · ".join(solvent_label(name) for name in available))
        missing = set(NATURAL_SOLVENTS) - {solvent_label(name) for name in available}
        if missing:
            st.warning("Not supported by the loaded models: " + ", ".join(sorted(missing)))
    smiles = st.text_area("Molecular SMILES", height=100, placeholder="e.g. CCO")
    left, right = st.columns(2)
    target = left.number_input("Target Rf", 0., 1., .50, .01, format="%.3f")
    tolerance = right.number_input("Tolerance (±)", 0., 1., 0., .01, format="%.3f")
    count = st.selectbox("Solvents per system", [1,2,3,4], index=1)
    st.caption("Ratios are searched in steps of 1%. Set tolerance to zero to search for a single target.")
    query = (smiles.strip(), target, tolerance, count)
    result_key = "natural_search_export_result" if natural_products else "search_export_result"
    if st.button("Find top 10 systems", type="primary"):
        with running("Scoring candidate mixtures…"):
            try:
                table, all_mixtures = search(ensemble, *query, natural_products=natural_products)
            except ValueError as exc:
                st.error(str(exc))
                return
        st.session_state[result_key] = (query, table, all_mixtures)
    result = st.session_state.get(result_key)
    if result and result[0] == query:
        _, table, all_mixtures = result
        st.subheader("Predicted solvent systems")
        st.caption(f"{len(all_mixtures):,} unique mixtures evaluated. Each solvent combination appears once, with its closest sampled ratio.")
        if tolerance:
            hits = int(table["Within range"].sum())
            st.write(f"{hits} of {len(table)} shown fall within Rf {max(0,target-tolerance):.3f}–{min(1,target+tolerance):.3f}.")
        table = table.copy()
        table.insert(0, "Rank", range(1,len(table)+1))
        st.dataframe(table, hide_index=True, column_config={c:st.column_config.NumberColumn(format="%.5f") for c in ["Predicted Rf","|ΔRf|"]})
        st.download_button("Download top 10 (CSV)",table.to_csv(index=False), "tlc_solvent_search.csv","text/csv")
        st.download_button(f"Download all {len(all_mixtures):,} evaluated mixtures (CSV)",
                           all_mixtures.to_csv(index=False), "tlc_all_evaluated_mixtures.csv", "text/csv")

