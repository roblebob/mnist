#!/usr/bin/env bash

# instructions taken from: https://research.google.com/colaboratory/local-runtimes.html







########################################################################################################################
# Step 2:
# Install and enable the jupyter_http_over_ws jupyter extension (one-time)
# The jupyter_http_over_ws extension is authored by the Colaboratory team and available on GitHub.

pip install --upgrade jupyter_http_over_ws
jupyter serverextension enable --py jupyter_http_over_ws



########################################################################################################################
# Step 3:
# Start server and authenticate
# New notebook servers are started normally, though you will need to set a flag to explicitly trust WebSocket connections from the Colaboratory frontend.

jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0

# Make note of the port that you start your Jupyter notebook server with as you'll need to provide this in the next step.