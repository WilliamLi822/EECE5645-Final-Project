def to_lower_case(s):
    """ Convert a string to lowercase. E.g., 'BaNaNa' becomes 'banana'. 
    """
    return s.lower()


def create_list_from_file(filename):
    """Read words from a file and convert them to a list.
   
    Input:
    - filename: The name of a file containing one word per line.
    
    Returns
    - wordlist: a list containing all the words in the file, as strings.

    """
    wordlist = []
    with open(filename) as f:
        line = f.readline()
        while line:
            wordlist.append(line.strip())
            line = f.readline()
        return wordlist     


def strip_non_alpha(s):
    """ Remove non-alphabetic characters from the beginning and end of a string. 

    E.g. ',1what?!"' should become 'what'. Non-alphabetic characters in the middle 
    of the string should not be removed. E.g. "haven't" should remain unaltered."""
    
    s = s.strip() # Remove spaces at the beginning and at the end of the string
    if not s:
        return s
    if not s.isalpha() and len(s)==1:
        return ''
    for start, char in enumerate(s):
        if char.isalpha():
            break
    for end, char in enumerate(s[::-1]):
        if char.isalpha():
            break
    return s[start:len(s) - end]

def is_link(s):
    """ Remove non-alphabetic characters from the beginning and end of a string. 

    E.g. ',1what?!"' should become 'what'. Non-alphabetic characters in the middle 
    of the string should not be removed. E.g. "haven't" should remain unaltered."""
    
    s = s.strip() # Remove spaces at the beginning and at the end of the string
    if not s:
        return s
    elif 'http' in s:
        return ''
    elif '@' in s:
        return '' 
    else: 
        return s


def is_inflection_of(s1,s2):
    """ Tests if s1 is a common inflection of s2. 

    The function first (a) converts both strings to lowercase and (b) strips
    non-alphabetic characters from the beginning and end of each string. 
    Then, it returns True if the two resulting two strings are equal, or 
    the first string can be produced from the second by adding the following
    endings:
    (a) 's
    (b) s
    (c) es
    (d) ing
    (e) ed
    (f) d
    """

    s1 = strip_non_alpha(to_lower_case(s1))
    s2 = strip_non_alpha(to_lower_case(s2))
    if s1 == s2:
        return True
    elif s2 == s1[:len(s2)] and s1[len(s2):] in ["'s", "s","es","ing","ed","d"]: 
        return True
    else:
        return False
    

def same(s1,s2):
    "Return True if one of the input strings is the inflection of the other."
    return is_inflection_of(s1,s2) or is_inflection_of(s2,s1)

def find_match(word,word_list):
    """Given a word, find a string in a list that is "the same" as this word.

    Input:
    - word: a string
    - word_list: a list of stings

    Return value:
    - A string in word_list that is "the same" as word, None otherwise.
    
    The string word is 'the same' as some string x in word_list, if word is the inflection of x,
    ignoring cases and leading or trailing non-alphabetic characters.
    """
    
    for word_list_element in word_list:
        if same(word,word_list_element):
            return word_list_element.strip()

    return None

if __name__=="__main__":
    
    # Test strip_non_alpha
    assert 'what'    == strip_non_alpha(',1what?!"')
    assert "haven't" == strip_non_alpha("haven't  ")
    assert 'wo=rld'  == strip_non_alpha('  ^(wo=rld"&')
    assert ''        == strip_non_alpha(',*&')
    assert ''        == strip_non_alpha(' ')


    # Test is_inflection_of and same
    assert is_inflection_of("1FriEnds","3^fRIend")
    assert is_inflection_of("reading","read")
    assert is_inflection_of("ed","&3^0")
    assert is_inflection_of("s",'  ')
    assert is_inflection_of("",'  ')
    assert not is_inflection_of(" ",'a')
    assert not is_inflection_of("1FriEnds","+$0")

    assert same("1FriEnds","3^fRIend")
    assert same("3^fRIend","1FriEnds")
    assert same("reading","read")
    assert same("read","reading")
    assert same("1FriEnds","3^fRIend")
    assert same(" ",'   ')
    assert not same(" ",'a')
    
    # Test find_match 
    word_list = ['friend','read','liked', '  ']
    assert 'friend' == find_match('friends',word_list)
    assert 'read'   == find_match('+reading^+',word_list)
    assert 'liked'  == find_match('like25',word_list)
    assert not find_match('%&hello',word_list)
    assert ''      == find_match('  ',word_list)
    assert ''      == find_match('45ed',word_list)
    assert not find_match('night^',word_list)
    
    pass
