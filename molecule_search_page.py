"""Streamlit page for finding COCONUT molecules by predicted retention factor."""

import os
from pathlib import Path

import streamlit as st

from ensemble import normalize_ratios
from molecule_search import EmbeddingDatabase, checkpoint_hashes
from solvent_rules import display_solvent, miscibility_flag


@st.cache_resource(show_spinner=False)
def _load_database(folder, metadata_signature, model_hashes, _ensemble):
    return EmbeddingDatabase(folder, _ensemble)


def render_molecule_search(ensemble):
    st.title("Find molecules")
    st.markdown(
        '<p class="intro">Find COCONUT structures with a predicted Rf close to your target.</p>',
        unsafe_allow_html=True,
    )
    st.caption("Matches use predicted Rf within ±0.10 of your target.")
    folder = Path(os.environ.get(
        "TLC_EMBEDDINGS_DIR", str(Path(__file__).resolve().parent / "coconut_embeddings"),
    ))
    if not (folder / "manifest.json").is_file() or not (folder / "summary.json").is_file():
        st.info("Place the complete coconut_embeddings folder beside app.py to enable molecule search.")
        return

    defaults = [name for name in ("n-Hexane", "Ethyl acetate") if name in ensemble.solvents]
    names = st.multiselect(
        "Mobile-phase solvents", ensemble.solvents, default=defaults,
        format_func=display_solvent, key="molecule_search_solvents",
    )
    parts = []
    for name in names:
        default = {"n-Hexane": 7.0, "Ethyl acetate": 3.0}.get(name, 1.0)
        parts.append(st.number_input(
            f"Parts of {display_solvent(name)}", min_value=0.0, value=default,
            step=0.1, key=f"molecule_search_parts_{name}",
        ))
    st.caption("Enter relative parts: for a 7:3 mixture, use 7 and 3.")
    target = st.number_input("Target Rf", 0.0, 1.0, 0.5, 0.01, key="molecule_search_target")
    tolerance = 0.1
    st.write(f"Match range: **{max(0, target - tolerance):.2f}–{min(1, target + tolerance):.2f}**")
    query = (tuple(names), tuple(parts), target)

    if st.button("Find matching molecules", type="primary", use_container_width=True):
        st.session_state.pop("molecule_search_result", None)
        progress = None
        try:
            ratio_text = ":".join(map(str, parts))
            mixture = normalize_ratios(names, ratio_text, ensemble.solvents)
            if miscibility_flag([name for name, fraction in mixture.items() if fraction > 0]) == "Check miscibility":
                st.warning("This solvent combination is flagged for a miscibility check.")
            signatures = tuple(
                (folder / name).stat().st_mtime_ns for name in ("manifest.json", "summary.json")
            )
            with st.spinner("Preparing molecule database..."):
                database = _load_database(
                    str(folder.resolve()), signatures,
                    tuple(sorted(checkpoint_hashes(ensemble).items())), ensemble,
                )
            progress = st.progress(0.0, text="Scoring database molecules...")

            def update(done, total):
                progress.progress(done / total, text=f"Scored {done:,} of {total:,} molecules")

            result = database.search(ensemble, names, ratio_text, target, tolerance, progress=update)
            st.session_state["molecule_search_result"] = (query, result)
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
        finally:
            if progress is not None:
                progress.empty()

    saved = st.session_state.get("molecule_search_result")
    if saved and saved[0] == query:
        result = saved[1]
        st.subheader("Matching molecules")
        st.caption(
            f"{len(result.table):,} matches from {result.searched:,} structures, "
            f"searched in {result.elapsed_seconds:.1f} seconds. Ranked by distance from the target Rf."
        )
        st.write("Mobile phase: " + " + ".join(
            f"{display_solvent(name)} {fraction:.1%}" for name, fraction in result.mixture.items()
        ))
        if result.table.empty:
            st.info("No predicted matches in this range. Try another target Rf or solvent mixture.")
            return
        if len(result.table) > 1000:
            st.caption("Showing the closest 1,000 matches. The CSV contains every matching molecule.")
        st.dataframe(
            result.table.head(1000), hide_index=True,
            column_order=["Rank", "COCONUT ID", "SMILES", "Predicted Rf", "Absolute Rf difference"],
            column_config={
                "Predicted Rf": st.column_config.NumberColumn(format="%.5f"),
                "Absolute Rf difference": st.column_config.NumberColumn(format="%.5f"),
            },
        )
        exported = result.table.assign(
            **{
                "Target Rf": result.target, "Tolerance": result.tolerance,
                "Mobile phase": " + ".join(
                    f"{display_solvent(name)} {fraction:.8%}" for name, fraction in result.mixture.items()
                ),
            }
        )
        st.download_button(
            "Download all matches (CSV)", exported.to_csv(index=False).encode("utf-8"),
            "coconut_rf_matches.csv", "text/csv",
        )
