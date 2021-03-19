#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   1970-01-01 01:00:00
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlande
# @Last Modified time: 2021-03-19 17:28:57


"""Ontonotes4.0 data loading script from .tsv file"""

from __future__ import absolute_import, division, print_function
import csv
import json
import os
import datasets


class Ontonotes(datasets.GeneratorBasedBuilder):
    """Onetonotes4.0 data"""

    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="Onetonotes4.0", version=4.0, description="Onetonotes4.0 dataset")
    ]

    # DEFAULT_CONFIG_NAME = "first_domain"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        features = datasets.Features(
            {
                "sent_id": datasets.Value("string"),
                "triplet": datasets.features.Sequence(
                    {
                        "id": datasets.Value("string"),
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

    def _split_generators(self, dl_manager, data):
        """Returns SplitGenerators."""
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data": data[int(0.0*len(data)):int(0.8*len(data))],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data": data[int(0.8*len(data)):int(0.9*len(data))],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data": data[int(0.9*len(data)):int(1.0*len(data))],
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """ Yields examples as (key, example) tuples. """

        data = []
        buffer = []
        sent_id = 0
        with open(filepath, encoding="utf-8") as f:
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
                        raw = " ".join(pair["token"] for item in buffer) 
                        data.append(
                            {
                                "sent_id": sent_id,
                                "triplet": buffer,
                                "raw": raw    
                            }
                            
                        )
                    yield {
                        "sent_id": sent_id,
                        "triplet": {
                            "id": buffer["id"],
                            "token": buffer["token"],
                            "pos_tag": buffer["pos_tag"]
                        },
                        "raw": raw
                    }