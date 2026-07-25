# GENERATION CODE FOR F0 AND F1 (ADAPTED FROM MARK AND TEGMARK'S GITHUB PAGE)
import random
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

ROOT = 'dataset'

df = pd.read_csv(os.path.join(ROOT, 'geonames.csv'))

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


def is_valid(row):
    name = row['ASCII Name']
    population = row['Population']
    country = row['Country name EN']
    
    # check that the country is valid
    if country not in countries:
        return False
    
    # check population is larger than 500000
    if not population > 500000:
        return False
    
    # check that not a city-state
    if name in country:
        return False
    
    # check there is no other city with the same name
    if len(df[df['ASCII Name'] == name]) > 1:
        return False
    
    return True

df = df[df.apply(is_valid, axis=1)]
df_out = {
    'statement' : [],
    'label' : [],
    'city' : [],
    'country' : [],
    'correct_country' : [],
}

neg_df_out = {
    'statement' : [],
    'label' : [],
    'city' : [],
    'country' : [],
    'correct_country' : [],
}

for _, row in df.iterrows():
    city = row['ASCII Name']
    country = countries[row['Country name EN']]

    # make true statement
    statement = f'The city of {city} is in {country}.'
    neg_statement = f'The city of {city} is not in {country}.'
    df_out['statement'].append(statement), neg_df_out['statement'].append(neg_statement)
    df_out['label'].append(1), neg_df_out['label'].append(0)
    df_out['city'].append(city), neg_df_out['city'].append(city)
    df_out['country'].append(country), neg_df_out['country'].append(country)
    df_out['correct_country'].append(country), neg_df_out['correct_country'].append(country)

    # make false statement
    false_country = countries[df['Country name EN'].sample(1).iloc[0]]
    while false_country == country:
        false_country = countries[df['Country name EN'].sample(1).iloc[0]]
    statement = f'The city of {city} is in {false_country}.'
    neg_statement = f'The city of {city} is not in {false_country}.'
    df_out['statement'].append(statement), neg_df_out['statement'].append(neg_statement)
    df_out['label'].append(0), neg_df_out['label'].append(1)
    df_out['city'].append(city), neg_df_out['city'].append(city)
    df_out['country'].append(false_country), neg_df_out['country'].append(false_country)
    df_out['correct_country'].append(country), neg_df_out['correct_country'].append(country)

df_out, neg_df_out = pd.DataFrame(df_out), pd.DataFrame(neg_df_out)

df_out_train, df_out_test = train_test_split(df_out, test_size=0.3, train_size=0.7, stratify=df_out["label"], shuffle=True, random_state=42)
neg_df_train, neg_df_test = train_test_split(neg_df_out, test_size=0.3, train_size=0.7, stratify=neg_df_out["label"], shuffle=True, random_state=42)

df_out_train.to_csv(os.path.join(ROOT, 'F0_train.csv'), index=False)
df_out_test.to_csv(os.path.join(ROOT, 'F0_test.csv'), index=False)
neg_df_train.to_csv(os.path.join(ROOT, 'F1_train.csv'), index=False)
neg_df_test.to_csv(os.path.join(ROOT, 'F1_test.csv'), index=False)