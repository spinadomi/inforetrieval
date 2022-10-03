##
# Simple linked list classes
#
# Nothing to see here folks
##
class Node:
  def __init__(self, data):
    self.item = data
    self.ref = None
    
class LinkedListIterator:
  def __init__(self, ll):
    self.current = ll.start_node
    
  def __iter__(self):
    return self
    
  def __next__(self):
    if not self.current:
      raise StopIteration
    else:
      item = self.current.item
      self.current = self.current.ref
      return item

class LinkedList:
  def __init__(self):
    # Currently Empty
    self.start_node = None
    self.tail_node = None

    # Current Size
    self.size = 0

  def insert_at_front(self, data):
    new_node = Node(data)
    new_node.ref = self.start_node
    
    if self.start_node is None:
      self.tail_node = new_node
      
    self.start_node = new_node
    self.size += 1

  def insert_at_end(self, data):
    new_node = Node(data)

	# Empty list?  Start/Tail point to same item then
    if self.start_node is None:
      self.start_node = new_node
      self.tail_node = new_node
      self.size += 1
      return

	# Extend end of list
    self.tail_node.ref = new_node
    self.tail_node = new_node
   
    self.size += 1

  # Function to search the list
  #
  # We make this flexible with an optional parameter for the user
  # to include their own comparison function
  #
  # WARNING:  This function returns the *FIRST* match regardless of
  #           method used.
  def search(self, data, search_func = None):
    n = self.start_node
    while n is not None:
      if search_func is None:
        if n.item == data:
          return n.item
      else:
        if search_func(n.item, data) is True:
          return n.item
      n = n.ref

    return None
    
  # Return as a python list - lazy function
  def get_as_list(self):
    l = [ ]
    
    if self.start_node is None:
      pass
    else:
      n = self.start_node
      while n is not None:
        l.append(n.item)
        n = n.ref
        
    return l
    
  # Get list length
  def get_size(self):
    return self.size

  # testing function
  def print_list(self):
    if self.start_node is None:
      print("List is empty.")
    else:
      n = self.start_node
      while n is not None:
        print(n.item)
        n = n.ref

