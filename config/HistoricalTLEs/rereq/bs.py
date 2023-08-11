# picking to rereq + moving them to new folder
# import os

# mypath = os.getcwd()

# rereq = []
# for filename in os.listdir(mypath):
#  	if filename[-3:] == "txt" and filename[0:3] == "sat":
# 	 	with open(os.path.join(mypath, filename), 'r') as f:
# 	 		text = f.readlines()
# 	 		if text[0].split()[3][0:5] != "19001":
# 	 			# print("Start Date is different:", filename, text[0].split()[3][0:5])
# 	 			rereq.append(filename[3:-4])
# 	 			# os.rename(mypath + "/" + filename, mypath + "/rereq/" + filename)
# 	 		elif text[-3].split()[3][0:5] != "22001" and text[-3].split()[3][0:5] != "21365":
# 	 			# print("End Date is different:", filename, text[-3].split()[3][0:5])
# 	 			rereq.append(filename[3:-4])
# 	 			# os.rename(mypath + "/" + filename, mypath + "/rereq/" + filename)
# print(len(rereq))
# # print(rereq)

# for each in rereq:
# 	with open('req.txt', 'a') as the_file:
# 	    the_file.write(each + '\n')

# writing to file
# import os

# mypath = os.getcwd()

# rereq = []
# for filename in os.listdir(mypath):
#  	if filename[-3:] == "txt" and filename[0:3] == "sat":
# 	 	rereq.append(filename[3:-4])
	 	
# print(len(rereq))
# # print(rereq)

# for each in rereq:
# 	with open('req.txt', 'a') as the_file:
# 	    the_file.write(each + '\n')

# checking
import csv
import os

mypath = os.getcwd() + "/../"

already = []
for filename in os.listdir(mypath):
 	if filename[-3:] == "txt" and filename[0:3] == "sat":
	 	already.append(filename[3:-4])
	 	
# print(len(already))

from pandas import *

data = read_csv(os.getcwd() + "/all.csv")
alls = data['norad'].tolist()
alls = [str(x) for x in alls]
# print(len(alls))
# print(type(alls[0]))

notin = []
for each in alls:
	if each in already:
		continue
	else:
		notin.append(each)

print((notin))
for each in notin:
	with open('notin.txt', 'a') as the_file:
	    the_file.write(each + '\n')