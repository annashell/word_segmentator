import nltk

str1 = "4010102012101020101010102040101010104010201010"
str2 = "401020120010210210102100201010210201"


possible_symb = ['0', '1', '2']


def delete_same_symbols(str_to_process):
    normalized = []
    previous = None
    for char in str_to_process:
        if char != previous:
            normalized.append(char)
        previous = char
    return ''.join(normalized)


def make_str_alignment(string_1, string_2, count=0):
    for i, ch in enumerate(list(string_1)):
        count += 1

        modifications_arr = []

        if ch == string_2[i] or ch == '4':
            continue

        levenst_dist = nltk.edit_distance(string_1.split(), string_2.split())

        if levenst_dist == string_1.lower().count('4'):
            return modifications_arr
        edit_variants = []

        str1_with_del_symb = delete_same_symbols(string_1[:i] + string_1[i:])
        levenstein_dist_del = nltk.edit_distance(str1_with_del_symb, string_2)
        edit_variants.append(("del", levenstein_dist_del))
        possible_symb_to_replace = [x for x in possible_symb if x != ch]
        for symb in possible_symb_to_replace:
            str1_with_repl_symb = delete_same_symbols(string_1[:i] + symb + string_1[i:])
            levenstein_dist_repl = nltk.edit_distance(str1_with_repl_symb, string_2)
            edit_variants.append((symb, levenstein_dist_repl))

        distances = [x[1] for x in edit_variants]
        best_var = edit_variants[distances.index(min(distances))]
        if best_var[0] == 'del':
            new_str = str1_with_del_symb
            modifications_arr.append(("-", count))
        else:
            new_str = delete_same_symbols(string_1[:i] + best_var[0] + string_1[i:])
            modifications_arr.append((best_var[0], count))
        return make_str_alignment(new_str, str2, count)


mods = make_str_alignment(str1, str2)
print(mods)




