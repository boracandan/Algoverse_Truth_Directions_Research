import csv
import os

ROOT = 'datasets/plain_dataset'

GEONAMES_COLUMNS = [
    'Geoname ID', 'Name', 'ASCII Name', 'Alternate Names', 'Latitude', 'Longitude',
    'Feature Class', 'Feature Code', 'Country Code', 'CC2', 'Admin1 Code',
    'Admin2 Code', 'Admin3 Code', 'Admin4 Code', 'Population', 'Elevation',
    'DEM', 'Timezone', 'Modification Date',
]

code_to_name = {}
with open(os.path.join(ROOT, 'CountryInfo.txt'), encoding='utf-8') as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.rstrip('\n').split('\t')
        code_to_name[parts[0]] = parts[4]

with open(os.path.join(ROOT, 'cities1000.txt'), encoding='utf-8') as fin, \
     open(os.path.join(ROOT, 'geonames.csv'), 'w', encoding='utf-8', newline='') as fout:
    writer = csv.writer(fout)
    writer.writerow(GEONAMES_COLUMNS + ['Country name EN'])
    for line in fin:
        row = line.rstrip('\n').split('\t')
        country_code = row[8]
        country_name = code_to_name.get(country_code, '')
        writer.writerow(row + [country_name])

print('wrote', os.path.join(ROOT, 'geonames.csv'))
