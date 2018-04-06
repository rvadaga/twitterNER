#!/bin/bash

INSTALL_COMMAND=apt-get

cd ../data
mkdir -p glove_embeddings cambridge_ltl stack_lstm_ner tagger_lample

PKG_OK=$(dpkg-query -W --showformat='${Status}\n' svn|grep "package installed")
echo -e "Checking for svn: \033[0;34m$PKG_OK\033[0m"
if [ "" == "$PKG_OK" ]; then
  echo -e "\033[0;31mNo svn. Setting up svn...\033[0m"
  sudo $INSTALL_COMMAND --yes install subversion
fi

svn checkout https://github.com/glample/tagger/trunk/dataset
mv dataset 01_conll_2003

cd cambridge_ltl
curl -OL http://yuca.test.iminds.be:8900/fgodin/downloads/word2vec_twitter_model.tar.gz
tar -xzvf word2vec_twitter_model.tar.gz

cd ../glove_embeddings
curl -OL http://nlp.stanford.edu/data/glove.6B.zip
curl -OL http://nlp.stanford.edu/data/glove.42B.300d.zip
curl -OL http://nlp.stanford.edu/data/glove.840B.300d.zip
curl -OL http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip -d glove.6B glove.6B.zip
unzip -d glove.42B.300d glove.42B.300d.zip
unzip -d glove.840B.300d glove.840B.300d.zip
unzip -d glove.twitter.27B glove.twitter.27B.zip

cd ../stack_lstm_ner
ggID='0B23ji47zTOQNUXYzbUVhdDN2ZmM'  
ggURL='https://drive.google.com/uc?export=download'  
filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"  
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"  
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${filename}"  

cd ../tagger_lample
ln -s ../stack_lstm_ner/sskip.100.vectors sskip.100.vectors
svn checkout https://github.com/glample/tagger/trunk/models

cd ../../scripts/