# CR-DDRDM
Proposed CR-DDRDM method reduces object/rating dimensions via CLSC &amp; DBSCAN, integrating PSO-optimized consensus for big rating data. Tested on 1,060 Booking.com hotels, it achieves 93.75% top-10 accuracy, outperforming TOPSIS/VIKOR. Offers novel approach for tourism management &amp; large-scale decision-making.
CR-DDRDM/
│── README.md          # Overview, setup, usage
│── requirements.txt   # Python dependencies
│── .gitignore         # Exclude .idea/, __pycache__/
│── data/              # (Optional) Sample dataset or instructions to download
│── src/
│   │── clustering/    # CLSC, DBSCAN code
│   │── ranking/       # TOPSIS, VIKOR, consensus logic
│   │── optimization/  # PSO implementation
│   │── main.py        # Entry point for experiments
│── results/           # (Optional) Benchmark comparisons
