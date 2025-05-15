#!/bin/bash

# VPS Connection details
VPS_USER=ubuntu
VPS_HOST=18.171.190.6
VPS_PORT=22
SSH_KEY=~/.ssh/LightsailDefaultKey-eu-west-2_2.pem

# Port configuration
FRONTEND_LOCAL_PORT=3000  # Next.js port
FRONTEND_REMOTE_PORT=3000  # Port to expose on VPS
BACKEND_LOCAL_PORT=5000   # Flask backend port
BACKEND_REMOTE_PORT=5000  # Port to expose on VPS

# Start the tunnel with autossh for both frontend and backend ports
autossh -M 0 -N \
  -o "ServerAliveInterval 30" \
  -o "ServerAliveCountMax 3" \
  -o "ExitOnForwardFailure yes" \
  -R ${FRONTEND_REMOTE_PORT}:localhost:${FRONTEND_LOCAL_PORT} \
  -R ${BACKEND_REMOTE_PORT}:localhost:${BACKEND_LOCAL_PORT} \
  ${VPS_USER}@${VPS_HOST} -i ${SSH_KEY} -p ${VPS_PORT} 