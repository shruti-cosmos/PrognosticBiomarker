"""Maps published comparator signature gene symbols to the exact
versioned Ensembl IDs present in the local gene expression matrix.
Run once before run_comparison.py."""

import json
import pandas as pd

import config

RAW_SIGNATURES = {
    "Zengin_2020_12gene": {
        "symbols": ["BCHE", "CCNA1", "CYP24A1", "DEPTOR", "MASP2", "MGLL",
                    "MYO1A", "PODXL2", "RAPGEF3", "SGK2", "TNNI2", "ZBTB16"],
        "coefficients_by_symbol": None,
        "reference": "Zengin & Onal-Suzek 2020, BMC Bioinformatics 21(Suppl 14):368",
    },
    "He_2020_13gene_metabolic": {
        "symbols": ["SLC2A1", "PCSK9", "KL", "ABCC2", "CAV3", "TCN1", "CDKN3",
                    "FFAR4", "CYP2F1", "SCN1A", "CYP4B1", "TK1", "TFAP2A"],
        "coefficients_by_symbol": None,  # refit: published coefficients assume a
                                          # different expression normalization
        "reference": "He et al. 2020, Mol Ther Oncolytics 19:265-277",
    },
    "Yang_2022_4gene_CCCRG": {
        "symbols": ["CCNB1", "CDC25C", "CENPM", "EXO1"],
        "coefficients_by_symbol": {
            "CCNB1": 0.131810530210757, "CDC25C": 0.0258950480925646,
            "CENPM": 0.0505775207458941, "EXO1": 0.0852753768507349,
        },
        "reference": "Yang et al. 2022, Front Genet 13:908104",
    },
}


def build_symbol_to_versioned_ensembl(data_columns):
    return {c.split(".")[0]: c for c in data_columns if c.startswith("ENSG")}


def main():
    print("Loading gene column headers...")
    data_cols = pd.read_csv(config.GENE_FILE, nrows=1).columns.tolist()
    bare_to_versioned = build_symbol_to_versioned_ensembl(data_cols)
    print(f"  {len(bare_to_versioned)} Ensembl gene columns found.")

    import mygene
    mg = mygene.MyGeneInfo()

    all_symbols = sorted({s for sig in RAW_SIGNATURES.values() for s in sig["symbols"]})
    print(f"Querying mygene for {len(all_symbols)} symbols...")
    res = mg.querymany(all_symbols, scopes="symbol", fields="ensembl.gene", species="human")

    symbol_to_bare_ensembl = {}
    for r in res:
        if "ensembl" not in r:
            print(f"  WARNING: no Ensembl hit for '{r['query']}' -- check manually.")
            continue
        ens = r["ensembl"]
        bare_id = ens[0]["gene"] if isinstance(ens, list) else ens["gene"]
        symbol_to_bare_ensembl[r["query"]] = bare_id

    resolved = {}
    for sig_name, sig in RAW_SIGNATURES.items():
        features_versioned, coefficients_versioned, missing = [], {}, []
        for sym in sig["symbols"]:
            bare = symbol_to_bare_ensembl.get(sym)
            versioned = bare_to_versioned.get(bare) if bare else None
            if versioned is None:
                missing.append(sym)
                continue
            features_versioned.append(versioned)
            if sig["coefficients_by_symbol"]:
                coefficients_versioned[versioned] = sig["coefficients_by_symbol"][sym]

        if missing:
            print(f"  [{sig_name}] MISSING from data: {missing} "
                  f"-- evaluated on remaining {len(features_versioned)}/{len(sig['symbols'])} genes only.")

        resolved[sig_name] = {
            "features": features_versioned,
            "coefficients": coefficients_versioned if sig["coefficients_by_symbol"] else None,
            "reference": sig["reference"],
        }

    with open("resolved_comparator_signatures.json", "w") as f:
        json.dump(resolved, f, indent=2)
    print("\nWrote resolved_comparator_signatures.json")


if __name__ == "__main__":
    main()
