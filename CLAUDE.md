This is a library of technical papers converted to Markdown, with a registry to keep track of them.

You'll either be asked to add a paper or search for one by topic, keywords, or more specific conditions.

# Records format

The records are maintaned in records.csv

It has the following fields:
- FOLDER: subfolder containing the paper, relative to the repo root.
Typically something like "ppo", "adam", etc.
This also serves as the unique identifier — no two papers can share a folder name.
- ARXIV_ID: arxiv id in the form XXXXXX.XXXXX (digit counts may vary)
- OPENREVIEW_ID:
- TITLE: lowercase english letters and spaces only.
only one space separates words.
- AUTHORS: same format as TITLE, with commas between authors.
- KEYWORDS: 20 keywords associated with the paper.
Should cover the problem, method, methodology, datasets, techniques, etc.
Only extract keywords from the abstract and introduction.
- COMMENT: potentially empty

# Adding to library

You'll be asked to add a paper to the library.
This will be either

- Add a paper "Title" to the library
- Add a paper with certain arxiv id/url
- Add a paper with certain openreview id/url
- Add a paper that is a pdf on disk, with pointer to it

You will download and parse the text, then update the records.

## Parsing text

Come up with a subfolder name.
For example, "adam" for the Adam paper, "ppo" for Proximal Policy Optimization.
Shorter is better, but check records.csv for conflicts.
If there's a conflict, pick a name that highlights the difference.

Follow these steps:

1. Create the folder (let's call it FOLDER), with subfolders FOLDER/pages_png, FOLDER/pages_md, and FOLDER/figures_and_tables.

2. Download the PDF into FOLDER, or copy it there if given a local path.

3. Convert each page to PNG using pdf2image (poppler), saving as FOLDER/pages_png/001.png, 002.png, etc.

4. Run subagents to convert each PNG page to Markdown, saving into FOLDER/pages_md with matching names.
Verify all completed in the main thread; relaunch any that failed.
Extract figures and tables as PNGs — crop them from the page images into FOLDER/figures_and_tables and reference them in the Markdown.
Ignore running headers, page numbers, and mini-headers when converting.
Finally, concatenate everything into FOLDER/paper.md.
After concatenating, verify all image paths are relative to FOLDER (where paper.md lives), not to the subfolders where they were originally written. Fix any that aren't.

5. Clean up by running `rm -r FOLDER/pages_png FOLDER/pages_md` — use exactly this command, no alternatives.
Just replace FOLDER with the actual name.

## Updating the records

Update records.csv so the new paper is searchable.
Fill out each field and append to records.csv;
Try to find the arxiv and openreview IDs when possible.

Run check_records.py and fix any errors it reports.
Only the last line.
If it flags suspiciously similar papers, read both and compare whether they're the same.
Run a subagent for that to avoid overfilling the context.
If they are the same, alert the user and ask whether to remove the new entry.
If they're different versions of the same paper, note that in the COMMENT field.
When referring to papers in the COMMENT, always use their FOLDER as the identifier.

# Querying library

You'll be asked to find papers in the library matching certain conditions.
Return all papers that satisfy them.
For example

- Find papers that study alternative positional embeddings in transformers.
- Find papers that have keywords policy exploration
- Are there any papers in the library that study importance of clipping threshold in ppo for RLHF?

First, identify keywords relevant to the query, then search records.csv to narrow down to up to 10 candidates.
Then run parallel agents on the candidates — each reads the full Markdown and evaluates whether the paper matches.
Read carefully the parts related to the query — e.g. if the query asks about a nuanced aspect of the paper, dig into the sections that cover it.
Don't limit your reasoning to the abstract and conclusion.
Check figures and tables that may be relevant.
Also look for papers that contradict the condition — e.g. if looking for papers showing weight decay is important in SFT of LLMs, and one says the opposite, include it in the results with a note that it contradicts.
