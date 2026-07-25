import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
import os

ROOT = 'dataset'

countries = {
    'Kenya' : 'Kenya',
    'Croatia' : 'Croatia',
    'Austria' : 'Austria',
    'Peru' : 'Peru',
    'Zambia' : 'Zambia',
    'Tajikistan' : 'Tajikistan',
    'Niger' : 'Niger',
    'Democratic Republic of the Congo' : 'the Democratic Republic of the Congo',
    'Algeria' : 'Algeria',
    'Trinidad and Tobago' : 'Trinidad and Tobago',
    'Cyprus' : 'Cyprus',
    'Mauritania' : 'Mauritania',
    'Uruguay' : 'Uruguay',
    'Slovenia' : 'Slovenia',
    'Saint Vincent and the Grenadines' : 'Saint Vincent and the Grenadines',
    'Bolivia' : 'Bolivia',
    'Malawi' : 'Malawi',
    'Bangladesh' : 'Bangladesh',
    'Turkey' : 'Turkey',
    'Vanuatu' : 'Vanuatu',
    'Madagascar' : 'Madagascar',
    'Hungary' : 'Hungary',
    'Kyrgyzstan' : 'Kyrgyzstan',
    'New Zealand' : 'New Zealand',
    'Uzbekistan' : 'Uzbekistan',
    'Iran' : 'Iran',
    'Togo' : 'Togo',
    'Tonga' : 'Tonga',
    'Monaco' : 'Monaco',
    'Luxembourg' : 'Luxembourg',
    'Costa Rica' : 'Costa Rica',
    'Belize' : 'Belize',
    'Montenegro' : 'Montenegro',
    'Spain' : 'Spain',
    'Lebanon' : 'Lebanon',
    'Poland' : 'Poland',
    'Tanzania' : 'Tanzania',
    'Australia' : 'Australia',
    'Angola' : 'Angola',
    'Saint Kitts and Nevis' : 'Saint Kitts and Nevis',
    'Panama' : 'Panama',
    'Samoa' : 'Samoa',
    'Switzerland' : 'Switzerland',
    'Burundi' : 'Burundi',
    'Mozambique' : 'Mozambique',
    'Papua New Guinea' : 'Papua New Guinea',
    'Bulgaria' : 'Bulgaria',
    'Chile' : 'Chile',
    'Mali' : 'Mali',
    'Venezuela' : 'Venezuela',
    'South Africa' : 'South Africa',
    'United Kingdom' : 'the United Kingdom',
    'Comoros' : 'Comoros',
    'Japan' : 'Japan',
    'Albania' : 'Albania',
    'Senegal' : 'Senegal',
    'Guatemala' : 'Guatemala',
    'Guinea' : 'Guinea',
    'Malaysia' : 'Malaysia',
    'Yemen' : 'Yemen',
    'Nauru' : 'Nauru',
    'Syria' : 'Syria',
    'Slovakia' : 'Slovakia',
    'Germany' : 'Germany',
    'Ecuador' : 'Ecuador',
    'Lithuania' : 'Lithuania',
    'Dominica' : 'Dominica',
    'Azerbaijan' : 'Azerbaijan',
    'Sudan' : 'Sudan',
    'Seychelles' : 'Seychelles',
    'Kiribati' : 'Kiribati',
    'Iraq' : 'Iraq',
    'Namibia' : 'Namibia',
    'Republic of the Congo' : 'the Republic of the Congo',
    'Andorra' : 'Andorra',
    'Canada' : 'Canada',
    'South Korea' : 'South Korea',
    'Bahamas' : 'the Bahamas',
    'Sierra Leone' : 'Sierra Leone',
    'Brazil' : 'Brazil',
    'Finland' : 'Finland',
    'Ukraine' : 'Ukraine',
    'Norway' : 'Norway',
    'Russia' : 'Russia',
    'Cuba' : 'Cuba',
    'Sao Tome and Principe' : 'Sao Tome and Principe',
    'Estonia' : 'Estonia',
    'Portugal' : 'Portugal',
    'Greece' : 'Greece',
    'Bhutan' : 'Bhutan',
    'Latvia' : 'Latvia',
    'Central African Republic' : 'the Central African Republic',
    'Zimbabwe' : 'Zimbabwe',
    'Lesotho' : 'Lesotho',
    'Moldova' : 'Moldova',
    'Mauritius' : 'Mauritius',
    'Palau' : 'Palau',
    'Nicaragua' : 'Nicaragua',
    'Djibouti' : 'Djibouti',
    'Gabon' : 'Gabon',
    'Dominican Republic' : 'the Dominican Republic',
    'Qatar' : 'Qatar',
    'Bosnia and Herzegovina' : 'Bosnia and Herzegovina',
    'Kazakhstan' : 'Kazakhstan',
    'Maldives' : 'the Maldives',
    'China' : 'China',
    'Vietnam' : 'Vietnam',
    'North Korea' : 'North Korea',
    'Myanmar' : 'Myanmar',
    'Turkmenistan' : 'Turkmenistan',
    'Barbados' : 'Barbados',
    'San Marino' : 'San Marino',
    'Romania' : 'Romania',
    'Armenia' : 'Armenia',
    'United Arab Emirates' : 'the United Arab Emirates',
    'Malta' : 'Malta',
    'Uganda' : 'Uganda',
    'United States' : 'the United States',
    'Saudi Arabia' : 'Saudi Arabia',
    'Ethiopia' : 'Ethiopia',
    'Guyana' : 'Guyana',
    'Benin' : 'Benin',
    'India' : 'India',
    'North Macedonia' : 'Macedonia',
    'Philippines' : 'the Philippines',
    'Mexico' : 'Mexico',
    'Fiji' : 'Fiji',
    'Bahrain' : 'Bahrain',
    'Belarus' : 'Belarus',
    'Afghanistan' : 'Afghanistan',
    'Ivory Coast' : "Côte d'Ivoire",
    'France' : 'France',
    'Kuwait' : 'Kuwait',
    'Czechia' : 'the Czech Republic',
    'Egypt' : 'Egypt',
    'Jordan' : 'Jordan',
    'Gambia' : 'the Gambia',
    'Equatorial Guinea' : 'Equatorial Guinea',
    'Oman' : 'Oman',
    'Denmark' : 'Denmark',
    'Haiti' : 'Haiti',
    'El Salvador' : 'El Salvador',
    'Liberia' : 'Liberia',
    'Tuvalu' : 'Tuvalu',
    'Burkina Faso' : 'Burkina Faso',
    'Chad' : 'Chad',
    'Guinea-Bissau' : 'Guinea-Bissau',
    'Cabo Verde' : 'Cape Verde',
    'Somalia' : 'Somalia',
    'Indonesia' : 'Indonesia',
    'Tunisia' : 'Tunisia',
    'Belgium' : 'Belgium',
    'Liechtenstein' : 'Liechtenstein',
    'Colombia' : 'Colombia',
    'Laos' : 'Laos',
    'Timor Leste' : 'Timor-Leste',
    'Honduras' : 'Honduras',
    'Italy' : 'Italy',
    'Serbia' : 'Serbia',
    'The Netherlands' : 'the Netherlands',
    'Jamaica' : 'Jamaica',
    'Eritrea' : 'Eritrea',
    'Nepal' : 'Nepal',
    'Eswatini' : 'Swaziland',
    'Antigua and Barbuda' : 'Antigua and Barbuda',
    'Rwanda' : 'Rwanda',
    'Paraguay' : 'Paraguay',
    'Sri Lanka' : 'Sri Lanka',
    'Iceland' : 'Iceland',
    'Morocco' : 'Morocco',
    'Suriname' : 'Suriname',
    'Argentina' : 'Argentina',
    'Mongolia' : 'Mongolia',
    'Botswana' : 'Botswana',
    'Thailand' : 'Thailand',
    'Cameroon' : 'Cameroon',
    'Ireland' : 'Ireland',
    'Nigeria' : 'Nigeria',
    'Cambodia' : 'Cambodia',
    'Sweden' : 'Sweden',
    'Pakistan' : 'Pakistan',
    'Ghana' : 'Ghana',
    'Singapore' : 'Singapore',
}

random.seed(42)
np.random.seed(42)

TRUE_SAMPLE_AMOUNT = 853
FALSE_SAMPLE_AMOUNT = 853

F0_df = pd.concat([pd.read_csv("dataset/F0_train.csv"), pd.read_csv("dataset/F0_test.csv")], axis=0)

F0_true_df = F0_df[F0_df["label"] == 1]
F0_false_df = F0_df[F0_df["label"] == 0]

# Building F2 from F0

def sample_stmt(df):
    row = df.sample(n=1).iloc[0]
    return row["statement"], row.name, row["city"]  # text, and its index as an ID

def make_f2_examples(F0_true_df, F0_false_df, n_true, n_false):
    df_out = {"statement": [], "label": [], "stmt1_id": [], "stmt2_id": []}
    seen = set()

    def add_example(pool1, pool2, label):
        while True:
            s1_text, s1_id, s1_city = sample_stmt(pool1)
            s2_text, s2_id, s2_city = sample_stmt(pool2)
            if s1_id == s2_id or s1_city == s2_city:
                continue
            new_statement = f"It is the case both that {s1_text} and that {s2_text}."
            if new_statement in seen:
                continue
            seen.add(new_statement)
            df_out["statement"].append(new_statement)
            df_out["label"].append(label)
            df_out["stmt1_id"].append(s1_id)
            df_out["stmt2_id"].append(s2_id)
            return

    # True-True -> label 1
    for _ in range(n_true):
        add_example(F0_true_df, F0_true_df, label=1)

    # False examples, split evenly across the three false-producing combinations
    n_per_false_type = n_false // 3
    remainder = n_false - (n_per_false_type * 3)  # handle rounding

    for _ in range(n_per_false_type + remainder):  # True-False
        add_example(F0_true_df, F0_false_df, label=0)
    for _ in range(n_per_false_type):  # False-True
        add_example(F0_false_df, F0_true_df, label=0)
    for _ in range(n_per_false_type):  # False-False
        add_example(F0_false_df, F0_false_df, label=0)

    return df_out

df_out_f2 = pd.DataFrame(make_f2_examples(F0_true_df, F0_false_df, TRUE_SAMPLE_AMOUNT, FALSE_SAMPLE_AMOUNT))

df_out_train_f2, df_out_test_f2 = train_test_split(df_out_f2, test_size=0.3, train_size=0.7, stratify=df_out_f2["label"], shuffle=True, random_state=42)

df_out_train_f2.to_csv(os.path.join(ROOT, 'F2_train.csv'), index=False)
df_out_test_f2.to_csv(os.path.join(ROOT, 'F2_test.csv'), index=False)


# Building F3 from F0

N_CITIES = 2  # F3 uses 2 cities
k_values = [0, 1, 2]

def make_f3_example_fixed_k(F0_true_df, target_label, stated_k):
    """
    target_label: 1 for True example, 0 for False example
    """
    country = random.choice(list(countries.values()))
    country_cities = F0_true_df[F0_true_df["country"] == country]
    other_cities = F0_true_df[F0_true_df["country"] != country]

    if target_label == 1:
        # True: stated_k must equal actual match count -> sample exactly stated_k matches
        actual_k = stated_k
    else:
        # False: actual match count must differ from stated_k
        possible_actual = [k for k in k_values if k != stated_k]
        actual_k = random.choice(possible_actual)

    if len(country_cities) < actual_k or len(other_cities) < (N_CITIES - actual_k):
        return None  # not enough cities available, caller should retry

    matches = country_cities.sample(n=actual_k)
    distractors = other_cities.sample(n=N_CITIES - actual_k)

    city_list = list(matches["city"]) + list(distractors["city"])
    random.shuffle(city_list)  # avoid positional shortcut (matches always first)

    city_str = ", ".join(city_list)
    statement = f"Exactly {stated_k} of the following cities are in {country}: {city_str}."

    return {
        "statement": statement,
        "label": target_label,
        "country": country,
        "cities": tuple(sorted(city_list)),
        "stated_k": stated_k,
        "actual_k": actual_k  # group ID for later leakage-safe splitting
    }

def make_f3_dataset(F0_true_df, n_true, n_false):
    rows = []
    seen = set()
    n_per_k_true = n_true // len(k_values)
    n_per_k_false = n_false // len(k_values)

    def generate_for_k(target_label, stated_k, n):
        count = 0
        while count < n:
            ex = make_f3_example_fixed_k(F0_true_df, target_label, stated_k)
            if ex is None or ex["cities"] in seen:
                continue
            seen.add(ex["cities"])
            rows.append(ex)
            count += 1

    for k in k_values:
        generate_for_k(1, k, n_per_k_true)
        generate_for_k(0, k, n_per_k_false)

    return rows

f3_df = pd.DataFrame(make_f3_dataset(F0_true_df, TRUE_SAMPLE_AMOUNT, FALSE_SAMPLE_AMOUNT))

f3_df_train, f3_df_test = train_test_split(f3_df, test_size=0.3, train_size=0.7, stratify=f3_df["label"], shuffle=True, random_state=42)

f3_df_train.to_csv(os.path.join(ROOT, 'F3_train.csv'), index=False)
f3_df_test.to_csv(os.path.join(ROOT, 'F3_test.csv'), index=False)

# Building F4 from F0

N_CITIES = 5
k_values = [0, 1, 2, 3, 4, 5]

def make_f4_example_fixed_k(F0_true_df, target_label, stated_k):
    """
    target_label: 1 for True example, 0 for False example
    """
    country = random.choice(list(countries.values()))
    country_cities = F0_true_df[F0_true_df["country"] == country]
    other_cities = F0_true_df[F0_true_df["country"] != country]

    if target_label == 1:
        # True: stated_k must equal actual match count -> sample exactly stated_k matches
        actual_k = stated_k
    else:
        # False: actual match count must differ from stated_k
        possible_actual = [k for k in k_values if k != stated_k]
        actual_k = random.choice(possible_actual)

    if len(country_cities) < actual_k or len(other_cities) < (N_CITIES - actual_k):
        return None  # not enough cities available, caller should retry

    matches = country_cities.sample(n=actual_k)
    distractors = other_cities.sample(n=N_CITIES - actual_k)

    city_list = list(matches["city"]) + list(distractors["city"])
    random.shuffle(city_list)  # avoid positional shortcut (matches always first)

    city_str = ", ".join(city_list)
    statement = f"Exactly {stated_k} of the following cities are in {country}: {city_str}."

    return {
        "statement": statement,
        "label": target_label,
        "country": country,
        "cities": tuple(sorted(city_list)),
        "stated_k": stated_k,
        "actual_k": actual_k   # group ID for later leakage-safe splitting
    }

def make_f4_dataset(F0_true_df, n_true, n_false):
    rows = []
    seen = set()
    n_per_k_true = n_true // len(k_values)
    n_per_k_false = n_false // len(k_values)

    def generate_for_k(target_label, stated_k, n):
        count = 0
        attempts = 0
        while count < n:
            attempts += 1
            if attempts > n * 50:  # safety valve
                print(f"WARNING: giving up on stated_k={stated_k}, label={target_label} after {attempts} attempts, only got {count}/{n}")
                break
            ex = make_f4_example_fixed_k(F0_true_df, target_label, stated_k)  
            if ex is None or ex["cities"] in seen:
                continue
            seen.add(ex["cities"])
            rows.append(ex)
            count += 1

    for k in k_values:
        generate_for_k(1, k, n_per_k_true)
        generate_for_k(0, k, n_per_k_false)

    return rows

f4_df = pd.DataFrame(make_f4_dataset(F0_true_df, TRUE_SAMPLE_AMOUNT, FALSE_SAMPLE_AMOUNT))

f4_df_train, f4_df_test = train_test_split(f4_df, test_size=0.3, train_size=0.7, stratify=f4_df["label"], shuffle=True, random_state=42)

f4_df_train.to_csv(os.path.join(ROOT, 'F4_train.csv'), index=False)
f4_df_test.to_csv(os.path.join(ROOT, 'F4_test.csv'), index=False)

# Building F5 from F0

N_CITIES = 6
K_RANGE = range(1, 6)  # 1..5

# All valid (k1, k2) pairs where k1 + k2 <= N_CITIES
valid_k1_k2_pairs = [(k1, k2) for k1 in K_RANGE for k2 in K_RANGE if k1 + k2 <= N_CITIES]

def make_f5_example(F0_true_df, target_label, stated_k1, stated_k2):
    """
    target_label: 1 for True example, 0 for False example
    stated_k1, stated_k2: now passed in directly, not sampled internally
    """
    country1, country2 = random.sample(list(countries.values()), 2)
    country1_cities = F0_true_df[F0_true_df["country"] == country1]
    country2_cities = F0_true_df[F0_true_df["country"] == country2]
    other_cities = F0_true_df[~F0_true_df["country"].isin([country1, country2])]

    if target_label == 1:
        actual_k1 = stated_k1
        actual_k2 = stated_k2
    else:
        # False: (actual_k1, actual_k2) must differ from (stated_k1, stated_k2)
        # as a PAIR -- not necessarily each individually different
        possible_actual_pairs = [
            (k1, k2) for (k1, k2) in valid_k1_k2_pairs
            if (k1, k2) != (stated_k1, stated_k2)
        ]
        actual_k1, actual_k2 = random.choice(possible_actual_pairs)

    if (len(country1_cities) < actual_k1
        or len(country2_cities) < actual_k2
        or len(other_cities) < (N_CITIES - actual_k1 - actual_k2)):
        return None  # not enough cities available, caller should retry

    matches1 = country1_cities.sample(n=actual_k1)
    matches2 = country2_cities.sample(n=actual_k2)
    distractors = other_cities.sample(n=N_CITIES - actual_k1 - actual_k2)

    city_list = list(matches1["city"]) + list(matches2["city"]) + list(distractors["city"])
    random.shuffle(city_list)

    city_str = ", ".join(city_list)
    statement = f"Exactly {stated_k1} of the following cities are in {country1} and {stated_k2} in {country2}: {city_str}."

    return {
        "statement": statement,
        "label": target_label,
        "countries": (country1, country2),
        "cities": tuple(sorted(city_list)),
        "stated_k1": stated_k1,
        "stated_k2": stated_k2,
        "actual_k1": actual_k1,
        "actual_k2": actual_k2,
    }

def make_f5_dataset(F0_true_df, n_true, n_false):
    rows = []
    seen = set()

    n_per_pair_true = n_true // len(valid_k1_k2_pairs)
    n_per_pair_false = n_false // len(valid_k1_k2_pairs)

    def generate(target_label, stated_k1, stated_k2, n):
        count = 0
        while count < n:
            ex = make_f5_example(F0_true_df, target_label, stated_k1, stated_k2)
            if ex is None or ex["cities"] in seen:
                continue
            seen.add(ex["cities"])
            rows.append(ex)
            count += 1
        if count < n:
            print(f"WARNING: only generated {count}/{n} for stated_k1={stated_k1}, "
                  f"stated_k2={stated_k2}, label={target_label}")

    for (k1, k2) in valid_k1_k2_pairs:
        generate(1, k1, k2, n_per_pair_true)
        generate(0, k1, k2, n_per_pair_false)

    return rows

f5_df = pd.DataFrame(make_f5_dataset(F0_true_df, TRUE_SAMPLE_AMOUNT, FALSE_SAMPLE_AMOUNT))

f5_df_train, f5_df_test = train_test_split(f5_df, test_size=0.3, train_size=0.7, stratify=f5_df["label"], shuffle=True, random_state=42)

f5_df_train.to_csv(os.path.join(ROOT, 'F5_train.csv'), index=False)
f5_df_test.to_csv(os.path.join(ROOT, 'F5_test.csv'), index=False)