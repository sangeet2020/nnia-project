import re
from collections import defaultdict 

#include the sample.conll file in the working directory, as file paths are relative
conll_file = open('sample.conll','r')
Lines = conll_file.readlines()

#collect positions of words, words and tags from the sample.conll file
words_tags_list = []
temp = []
words_count = 0
for line in Lines:
    temp = re.split(r'\s+', line)
    if '#' not in temp[0]: # ignore the lines starting with #  
        if temp[0] != '': #not an empty line, add position, word, tag
            words_tags_list.append([temp[2],temp[3],temp[4]])
            words_count += 1    #words counts tags per word percentages
        else:
            words_tags_list.append(['*'])
            
# write the important information of the data [position, word, tag] to 'sample.tsv'
f = open("sample.tsv", "w")
for line in words_tags_list:
    if line[0] != '*':
        f.write(line[0]+"\t"+line[1]+"\t"+line[2]+"\n")
    else:
        f.write(line[0]+"\n")
f.close()

# collect information about the data and form a dictionary of tags
max_len = 0
min_len = 100
acc_seq = 0
num_seq = 0
tags_list = defaultdict(list)
for i in range(len(words_tags_list)):
    seq_length = 0
    if words_tags_list[i] == ['*']:
        num_seq += 1 
        seq_length = int(words_tags_list[i-1][0]) + 1 #add 1 since it's 0-indexed
        acc_seq += seq_length             # accumulte seqs length to calculate Mean
        if seq_length > max_len:          #get the maximum length of sequences
            max_len = seq_length
        if seq_length < min_len:          #get the minimum length of sequnces
            min_len = seq_length
    else:
        if words_tags_list[i][2]  in tags_list:
            tags_list[words_tags_list[i][2]]+= 1    #get tags number for each tag
        else:
            tags_list[words_tags_list[i][2]] = 0

            
# get percentage of the words that have these tags.
for key in tags_list.keys():
    tags_list[key] = round(tags_list[key]/ words_count,2)*100   

#write the inforamtion above on sample.info
file_info = open("sample.info", "w")
file_info.writelines("Max sequence length: " + str(max_len) + '\n')
file_info.writelines("Min sequence length: "+str(min_len) + '\n')
file_info.writelines("Mean sequnce length"+ str(acc_seq/num_seq)+ '\n')
file_info.writelines("Number of sequnces: "+str(num_seq)+ '\n')
file_info.writelines("Tags: \n" +str(tags_list)+ '\n')
file_info.close()