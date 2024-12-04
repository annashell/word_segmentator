import itertools

import nltk


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
    if string_1 == string_2:
        return count

    for i, ch in enumerate(list(string_1)):

        if ch == string_2[i]:
            continue

        edit_variants = []

        str1_with_del_symb = delete_same_symbols(string_1[:i] + string_1[i:])
        levenstein_dist_del = nltk.edit_distance(str1_with_del_symb, string_2)

        edit_variants.append(("del", levenstein_dist_del))

        possible_symb_to_replace = [x for x in possible_symb if x != ch]
        for symb in possible_symb_to_replace:
            str1_with_repl_symb = delete_same_symbols(string_1[:i] + symb + string_1[i:])
            levenstein_dist_repl = nltk.edit_distance(str1_with_repl_symb, string_2)
            edit_variants.append(("r", levenstein_dist_repl))

        for symb in possible_symb_to_replace:
            str1_with_inserted_symb = delete_same_symbols(string_1[:i + 1] + symb + string_1[i:])
            levenstein_dist_repl = nltk.edit_distance(str1_with_inserted_symb, string_2)
            edit_variants.append(("i", levenstein_dist_repl))

        distances = [x[1] for x in edit_variants]
        best_var = edit_variants[distances.index(min(distances))]
        if best_var[0] == 'del':
            new_str = str1_with_del_symb
        elif best_var[1] == 'r':
            new_str = delete_same_symbols(string_1[:i] + best_var[0] + string_1[i:])
        elif best_var[1] == 'i':
            new_str = delete_same_symbols(string_1[:i + 1] + best_var[0] + string_1[i:])
        count += 1
        return make_str_alignment(new_str, str2, count)


def partition(arr, n):
    # Генерируем все возможные разбиения массива на N частей
    result = []
    length = len(arr)

    # Получаем все возможные индексы для разбиений
    indices = range(2, length)

    # Генерируем все комбинации индексов для разбиений
    for cuts in itertools.combinations(indices, n - 1):
        cuts = (0,) + cuts + (length,)
        partitioned = [arr[cuts[i]:cuts[i + 1]] for i in range(n)]
        if len(partitioned[0]) < 2:
            continue

        for i, part in enumerate(partitioned):
            if i == 0:
                continue
            part.insert(0, partitioned[i - 1][-1])

        result.append(partitioned)
    return result


def find_syntagma_distribution_variants (ac_string: str, txt_clusters, word_bound_indexes):
    num_of_syntagmas = ac_string.count("4") # должна быть 4 в начале строки
    syntagma_distribution_variants = partition(word_bound_indexes, num_of_syntagmas)

    txt_cluster_distribution_variants = []
    for variant in syntagma_distribution_variants:
        text_variant = []
        for part in variant:
            part_distribution = []
            start = part[0] if part[0] == 0 else part[0] + 1
            end = part[-1]
            for cluster_1, cluster_2 in zip(txt_clusters, txt_clusters[1:]):
                if end < cluster_1[0]:
                    break
                if start > cluster_2[0]:
                    continue
                if cluster_1[0] <= start < cluster_2[0]:
                    part_distribution.append(cluster_1[1])
                if start < cluster_1[0] < end:
                    part_distribution.append(cluster_1[1])
                if cluster_1[0] <= end <= cluster_2[0]:
                    if part_distribution[-1] != cluster_1[1]:
                        part_distribution.append(cluster_1[1])
                    break
            text_variant.append("".join(str(num) for num in part_distribution))
        txt_cluster_distribution_variants.append(text_variant)

    return txt_cluster_distribution_variants, syntagma_distribution_variants


def make_most_probable_syntagma_distribution(ac_string: str, txt_clusters, word_bound_indexes):
    ac_syntagmas = [i for i in ac_string.split("4") if i]

    txt_cluster_distribution_variants, syntagma_distribution_variants = find_syntagma_distribution_variants(ac_string, txt_clusters, word_bound_indexes)

    lev_distances = []
    for variant in txt_cluster_distribution_variants:
        lev = 0
        for i, part in enumerate(variant):
            lev += make_str_alignment(part, ac_syntagmas[i])
        lev_distances.append(lev)

    best_var = lev_distances.index(min(lev_distances))

    best_syntagma_distribution = syntagma_distribution_variants[best_var]

    return best_syntagma_distribution




str1 = "4010102012101020101010102040101010104010201010"
str2 = "401020120010210210102100201010210201"

word_boundaries_ind = [0, 9, 11, 16, 22, 33, 35, 43, 52, 60, 62, 67, 73, 77, 80]
txt_clusters = [(0, 0), (2, 1), (3, 0), (12, 2), (13, 0), (17, 1), (18, 2), (19, 0), (20, 0), (23, 1), (24, 0), (26, 2), (26, 1), (27, 0), (30, 2), (31, 1), (32, 0), (34, 1), (36, 0), (38, 2), (38, 1), (39, 0), (44, 0), (47, 2), (48, 0), (55, 1), (56, 0), (63, 1), (64, 0), (68, 2), (68, 1), (69, 0), (75, 2), (76, 0), (83, 1)]

# mods = make_str_alignment(str1, str2)
# print(mods)


# make_most_probable_syntagma_distribution(str1, txt_clusters, word_boundaries_ind)


make_str_alignment("0120", "010")




