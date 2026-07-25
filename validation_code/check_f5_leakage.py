import re
import pandas as pd

# Single-country pattern: "Exactly N of the following cities are in X: c1, c2, ..."
SINGLE_RE = re.compile(
    r"Exactly (\d+) of the following cities are in (.+?): (.+)\."
)
# Dual-country pattern: "Exactly N of the following cities are in X and M in Y: c1, c2, ..."
DUAL_RE = re.compile(
    r"Exactly (\d+) of the following cities are in (.+?) and (\d+) in (.+?): (.+)\."
)


def parse_statement(statement):
    s = statement.strip()
    m = DUAL_RE.match(s)
    if m:
        k1, country1, k2, country2, cities_str = m.groups()
        cities = tuple(sorted(c.strip() for c in cities_str.split(",")))
        return cities
    m = SINGLE_RE.match(s)
    if m:
        k1, country1, cities_str = m.groups()
        cities = tuple(sorted(c.strip() for c in cities_str.split(",")))
        return cities
    return None


def load(path):
    df = pd.read_csv(path)
    city_sets = df["statement"].apply(parse_statement)
    n_unparsed = city_sets.isna().sum()
    if n_unparsed:
        print(f"WARNING: {n_unparsed} statements in {path} did not match the expected pattern")
    df = df[city_sets.notna()].copy()
    df["city_set"] = city_sets[city_sets.notna()]
    return df


def check_conflicting_labels(df, name):
    grouped = df.groupby("statement")["label"].nunique()
    conflicts = grouped[grouped > 1]
    print(f"[{name}] rows: {len(df)}, unique statements: {df['statement'].nunique()}")
    print(f"[{name}] statements with conflicting labels: {len(conflicts)}")
    if len(conflicts):
        print(df[df["statement"].isin(conflicts.index)].sort_values("statement").head(10))
    return conflicts


def check_train_test_overlap(train_df, test_df, name):
    train_sets = set(train_df["city_set"])
    test_sets = set(test_df["city_set"])
    overlap = train_sets & test_sets

    train_rows_leaked = train_df["city_set"].isin(overlap).sum()
    test_rows_leaked = test_df["city_set"].isin(overlap).sum()

    print(f"\n--- [{name}] Train/Test city-group overlap ---")
    print(f"Unique city groups in train: {len(train_sets)}")
    print(f"Unique city groups in test:  {len(test_sets)}")
    print(f"Overlapping city groups:     {len(overlap)}")
    print(f"% of test groups also in train: {100 * len(overlap) / max(len(test_sets), 1):.2f}%")
    print(f"Train rows touching a leaked group: {train_rows_leaked} / {len(train_df)} "
          f"({100 * train_rows_leaked / len(train_df):.2f}%)")
    print(f"Test rows touching a leaked group:  {test_rows_leaked} / {len(test_df)} "
          f"({100 * test_rows_leaked / len(test_df):.2f}%)")

    if overlap:
        example = next(iter(overlap))
        print("\nExample overlapping city group:", example)
        print("Matching rows in train:")
        print(train_df[train_df["city_set"] == example][["statement", "label"]])
        print("Matching rows in test:")
        print(test_df[test_df["city_set"] == example][["statement", "label"]])

    return overlap


def run_check(name, train_path, test_path):
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    train_df = load(train_path)
    test_df = load(test_path)

    check_conflicting_labels(train_df, f"{name}_train")
    check_conflicting_labels(test_df, f"{name}_test")
    check_train_test_overlap(train_df, test_df, name)


if __name__ == "__main__":
    run_check("F3", "../dataset/F3_train.csv", "../dataset/F3_test.csv")
    run_check("F4", "../dataset/F4_train.csv", "../dataset/F4_test.csv")
    run_check("F5", "../dataset/F5_train.csv", "../dataset/F5_test.csv")
