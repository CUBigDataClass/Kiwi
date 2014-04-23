import jellyfish

#def Golden_List(sku, min_frequency):
  #stuff here

def Correct_List(sku, word_list_in, min_frequency, max_distance):
  #stuff here

def Spell_Check(check_word, known_word, max_distance):
  diff = jellyfish.levenshtein_distance(check_word, known_word)
  print diff
  if diff <= max_distance:
    return known_word
  else:
    return check_word

