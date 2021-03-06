#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Zena Al Khalili, Sangeet Sagar
# @Date:   1970-01-01 01:00:00
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlande
# @Last Modified time: 2021-03-20 19:48:28


"""Ontonotes4.0 data loading script from CoNLL files"""

from __future__ import absolute_import, division, print_function
import csv
import json
import os
import glob
import pdb
import datasets



class Ontonotes(datasets.GeneratorBasedBuilder):
    """Onetonotes4.0 data"""

    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="Onetonotes4.0", version=4.0, description="Onetonotes4.0 dataset")
    ]
    
    def _info(self):
        features = datasets.Features(
            {
                "sent_id": datasets.Value("int32"),
                "triplet": datasets.features.Sequence(
                    {
                        "id": datasets.Value("int32"),
                        "token": datasets.Value("string"),
                        "pos_tag": datasets.Value("string")    
                    }                    
                    
                ),
                "raw": datasets.Value("string")
            }
            )
        
        return datasets.DatasetInfo(
            features=features,
            supervised_keys=None,
        )

    def _split_generators(self, data):
        """Returns SplitGenerators."""
        
        self.extract_data(filepath=self.config.data_files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                
                gen_kwargs={
                    "data": self.data[int(0.0*len(self.data)):int(0.8*len(self.data))],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data": self.data[int(0.8*len(self.data)):int(0.9*len(self.data))],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data": self.data[int(0.9*len(self.data)):int(1.0*len(self.data))],
                },
            ),
        ]

    
    def extract_data(self, filepath):
        self.data = []
        buffer = []
        sent_id = 0
        os.chdir(filepath)
        for file in glob.glob("*.gold_conll"):
            with open(file, encoding="utf-8") as f:
                for line in f:
                    if line.startswith('#'):
                        pass
                    else:
                        cols = line.strip("\n").split()
                        if len(cols) != 0:
                            buffer.append(
                                {
                                    "id": cols[2],
                                    "token": cols[3],
                                    "pos_tag": cols[4]    
                                }
                            )
                            
                        else:
                            sent_id += 1
                            raw = " ".join(item["token"] for item in buffer) 
                            self.data.append(
                                {
                                    "sent_id": sent_id,
                                    "triplet": buffer,
                                    "raw": raw    
                                }
                                
                            )
                            buffer = []

    
    def _generate_examples(self, data):
        """ Yields examples as (key, example) tuples. """
        
        for id, item in enumerate(data):
            yield id, {
                "sent_id": item["sent_id"],
                "triplet": {
                    "id": [i["id"] for i in item["triplet"]],
                    "token": [i["token"] for i in item["triplet"]],
                    "pos_tag": [i["pos_tag"] for i in item["triplet"]]
                },
                "raw": item["raw"]
            }
