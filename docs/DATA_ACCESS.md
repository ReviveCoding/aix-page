# Data access

KDD Cup 2012 Track 2 data are not redistributed in this repository. Obtain access independently through the official [Kaggle competition page](https://www.kaggle.com/competitions/kddcup2012-track2) and comply with its rules.

Place the downloaded archive at:

```text
<repo>/data/external/track2.zip
```

The archive used for this project had exactly 2,882,914,995 bytes and local SHA-256 `E1FBB172130BA9AE55593F74E5B6BFDD74E1E6046A0B540FFB6B07DD087F5686`. This is the checksum of this project's verified official Kaggle download; it is not represented as a Kaggle-published checksum.

Expected members include:

- `track2/training.txt`
- `track2/queryid_tokensid.txt`
- `track2/purchasedkeywordid_tokensid.txt`
- `track2/titleid_tokensid.txt`
- `track2/descriptionid_tokensid.txt`
- `track2/userid_profile.txt`

Verify, ingest by streaming the ZIP member, and build the DuckDB marts:

```powershell
.\.venv\Scripts\python.exe scriptserify_kdd_source.py
.\.venv\Scripts\python.exe scripts\ingest_kdd.py
.\.venv\Scripts\python.exe scripts\ingest_kdd_companions.py
.\.venv\Scripts\python.exe scriptsuild_kdd_marts.py
```

The parser enforces the official 12-field TSV schema. It never requires loading the 10.6 GB uncompressed training member into memory. Generated Parquet and DuckDB files are ignored by Git.
