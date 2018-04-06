#!/bin/bash

INSTALL_COMMAND=apt-get

cd ../projects
curl -L https://github.com/napsternxg/TwitterNER/archive/master.zip -o twitter_ner_napsternxg.zip
unzip twitter_ner_napsternxg.zip
mv TwitterNER-master 05_twitter_ner_napsternxg

cd ../scripts/