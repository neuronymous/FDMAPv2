#!/bin/bash
apt install -y g++
apt install -y libtool
apt install -y libxml2-dev
apt install -y make
cd ~
#tar -xvf madp-0.4.1.tar.gz
#tar -xvf exp-problems.tar.gz
#cp exp-problems/BPNEW* madp-0.4.1/problems/
cd madp-0.4.1
#./configure
sed -i -e "s/\$(CSTANDARD)/\$(CSTD11)/g" src/parser/Makefile
#ADD SED FOR PRINTING JOINT POLICY IN GMAA
#make
#mkdir results
mkdir ~/.madp
cd ~/.madp
ln -s ~/madp-0.4.1/problems
ln -s ~/madp-0.4.1/results
mkdir GMAA
mkdir JESP
