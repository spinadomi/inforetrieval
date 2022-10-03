#####
#
# UTILITY FUNCTIONS BELOW - Can safely ignore
#
#####

##
# Given folder name it scans for all files ending in ".txt".
#
# The contents of each file is read in as a single string.
#
# @param - Folder to find .txt files
#
# @return - List containing ordered pairs (a, b)
#           a - corresponds to document name
#           b - corresponds to the text of the document as a string
###
def read_data(folder):
  import os

  documents = [ ]
  for file in os.listdir(folder):
    if file.endswith(".txt"):
      documents.append((file.rsplit('.', 1)[0], (read_text_file_whole(folder + "/" + file))))

  return documents

##
# Reads in an entire text file as a giant string
#
# @param fileName - Name of file to read in.
#
# @return The contents of the file as a string.
#
# Revision History:
# ~~~~~~~~~~~~~~~~~
# 07/10/2019 - Created (CJL).
###
def read_text_file_whole(fileName):
  try:
    file_obj = open(fileName, 'r')
    fileContents = file_obj.read()
  except OSError as ex:
    print("In IOErrorBlock")
    print("In method [read_text_file_whole] - " + str(ex))
    print("Out IOErrorBlock")
    return ""

  file_obj.close()

  return fileContents