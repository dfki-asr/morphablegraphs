# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:11:52 2015

Simple Motion Graphs command line interface for pipeline tests.

@author: erhe01
"""


import os
 # change working directory to the script file directory
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname)
import glob
import time
import sys
import urllib.request, urllib.error, urllib.parse
import json
from anim_utils.utilities.io_helper_functions import load_json_file
SERVICE_CONFIG_FILE = "config" + os.sep + "service.config"
ALGORITHM_CONFIG_FILE = "config" + os.sep + "standard_algorithm.config"


def config_pipeline(service_config):
    """Sends the current config to the morphablegraphs server
    """

    mg_input = load_json_file(ALGORITHM_CONFIG_FILE)
    data = json.dumps(mg_input)
    mg_server_url = 'http://localhost:8888/config_morphablegraphs'
    request = urllib.request.Request(mg_server_url, data)
    
    print("send config and wait for motion generator...")
    handler = urllib.request.urlopen(request)
    result = handler.read()
    print(result)


def main():
    """Loads the latest file added to the input directory specified in
        service_config.json and runs the algorithm.
    """

    if os.path.isfile(SERVICE_CONFIG_FILE):
        service_config = load_json_file(SERVICE_CONFIG_FILE)

        config_pipeline(service_config)
    else:
        print("Error: Could not read service config file", SERVICE_CONFIG_FILE)


if __name__ == "__main__":
    """example call:
       mg_pipeline_interface.py
    """
    import warnings
    warnings.simplefilter("ignore")
    main()