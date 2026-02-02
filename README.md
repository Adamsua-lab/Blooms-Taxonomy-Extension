# Bloom-s-Taxonomy-extension
Bloom Taxonomy Verb Classification (BT1–BT4)

This repository contains a complete pipeline to extend and classify Bloom’s Taxonomy (Cognitive Domain) verb lists using an NLP + ML workflow. It includes:

A core Bloom verb list (grounded set)

Verb extraction from engineering PDFs (papers/books)

WordNet-based filtering/verification

A trained multi-label (one-vs-rest) classifier that assigns Bloom levels to verbs

Expert validation files for analysis and reporting

Project workflow at a glance

  BT1 → BT2 → BT3 → BT4

  1.BT1: Core verb list preparation / formatting and baseline checks

  2.BT2: Extract candidate verbs from engineering PDFs + filter using WordNet

  3.BT3: Train + tune multi-label classifier using the core list

  4.BT4: Apply best model to new verbs + export final labeled lists and expert-sampling files

  Respositry Structure:
  
.
├── BT1/ -> Core list setup
├── BT2/ -> PDF verb extraction + WordNet filtering
├── BT3/ -> Model training + selection
├── BT4/ -> Production labeling + exports
├── Paper & Books used for extraction/
│   └── (PDF files used in BT2)
├── best model saved/
│   └── (trained model artifacts + weights for reproducibility)
└── results/
    ├── BTdone_paper_list.txt
    ├── BTverblist_core.txt
    ├── BTverblist_new.txt
    ├── Verb List Classified by Model.csv
    ├── ExpertSample_single (Model Classified Sample).csv
    └── Final File(with model & Expert Classification).csv

Results folder (outputs)

All primary outputs are stored in results/:

  BTdone_paper_list.txt
  A verification log generated while processing PDFs in BT2 (helps confirm which PDFs were read/processed).
  
  BTverblist_core.txt
  The core verb list used for training (ground truth source list).
  
  BTverblist_new.txt
  The new/extended verbs extracted from PDFs and verified via WordNet.
  
  Verb List Classified by Model.csv
  Same content as BTverblist_new.txt, but includes the model-assigned Bloom level labels (and any extra columns your pipeline exports).
  
  ExpertSample_single (Model Classified Sample).csv
  A 30-verb sample (5 per Bloom level) generated for expert labeling.
  
  Final File(with model & Expert Classification).csv
  The combined file used for analysis: model labels + expert labels (used to compute agreement/statistics).

Best model (reproducibility)
Best Model Saved/

  Stores the final selected model and any supporting artifacts needed to reproduce predictions, such as:
  
  trained classifier weights
  
  calibration objects / thresholds (if your pipeline saves them)
  
  metadata (feature/label order, versioning, etc.)
  
  This folder is what you use when you want to run BT4 predictions without retraining.
