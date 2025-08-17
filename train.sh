#!/bin/sh
if [ $1 ]; then
	echo 'eval'
	uv run main.py -e -d mnist -a ca
	uv run main.py -e -d mnist -a faa
	uv run main.py -e -d cifar10 -a ca
	uv run main.py -e -d cifar10 -a faa
else
	echo 'train'
	uv run main.py -d mnist -a ca
	uv run main.py -d mnist -a faa
	uv run main.py -d cifar10 -a ca
	uv run main.py -d cifar10 -a faa
fi
