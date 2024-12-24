import itertools
from statistics import stdev

import nltk
import numpy as np
from scipy import stats

from utils.json_utils import get_object_from_json

possible_symb = ['0', '1', '2']


def delete_same_symbols(str_to_process):
    normalized = []
    previous = None
    for char in str_to_process:
        if char != previous:
            normalized.append(char)
        previous = char
    return ''.join(normalized)


def make_str_alignment(string_1, string_2, count=0, operations = []):
    if string_1 == string_2:
        return (count, operations)

    for i, ch in enumerate(list(string_1)):

        if string_2.startswith(string_1):
            return (len(string_2) - len(string_1) + count, operations)

        if string_1.startswith(string_2):
            return (len(string_1) - len(string_2) + count, operations)

        if ch == string_2[i]:
            continue

        edit_variants = []

        str1_with_del_symb = delete_same_symbols(string_1[:i] + string_1[i + 1:])
        levenstein_dist_del = nltk.edit_distance(str1_with_del_symb, string_2)

        edit_variants.append(("del", levenstein_dist_del))

        possible_symb_to_replace = [x for x in possible_symb if x != ch]
        for symb in possible_symb_to_replace:
            str1_with_repl_symb = delete_same_symbols(string_1[:i] + symb + string_1[i + 1:])
            levenstein_dist_repl = nltk.edit_distance(str1_with_repl_symb, string_2)
            edit_variants.append(("r", levenstein_dist_repl, symb))

        for symb in possible_symb_to_replace:
            str1_with_inserted_symb = delete_same_symbols(string_1[:i] + symb + string_1[i:])
            levenstein_dist_repl = nltk.edit_distance(str1_with_inserted_symb, string_2)
            edit_variants.append(("i", levenstein_dist_repl, symb))

        distances = [x[1] for x in edit_variants]
        best_var = edit_variants[distances.index(min(distances))]
        if best_var[0] == 'del':
            new_str = str1_with_del_symb
            operations.append(('d', i, 0))
        elif best_var[0] == 'r':
            new_str = delete_same_symbols(string_1[:i] + best_var[2] + string_1[i + 1:])
            operations.append(('r', i, best_var[2]))
        elif best_var[0] == 'i':
            new_str = delete_same_symbols(string_1[:i] + best_var[2] + string_1[i:])
            operations.append(('i', i, best_var[2]))
        count += 1
        return make_str_alignment(new_str, string_2, count, operations)


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


def find_syntagma_distribution_variants(ac_string: str, txt_clusters, word_bound_indexes, txt_arr):
    num_of_syntagmas = ac_string.count("4")  # должна быть 4 в начале строки
    syntagma_distribution_variants = partition(word_bound_indexes, num_of_syntagmas)

    text_syntagmas = []

    txt_cluster_distribution_variants = []
    for variant in syntagma_distribution_variants:
        text_variant = []
        transcrip_variant = []
        for part in variant:
            text_str = txt_arr[part[0]: part[-1]]
            transcrip_variant.append(text_str)

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
        text_syntagmas.append(transcrip_variant)

    return txt_cluster_distribution_variants, syntagma_distribution_variants, text_syntagmas


def make_most_probable_syntagma_distribution(ac_string: str, txt_clusters, word_bound_indexes, ac_synt_durations,
                                             txt_arr):
    ac_syntagmas = [i for i in ac_string.split("4") if i]

    txt_cluster_distribution_variants, syntagma_distribution_variants, text_syntagmas = find_syntagma_distribution_variants(
        ac_string,
        txt_clusters,
        word_bound_indexes, txt_arr)
    print("Найдены варианты разбиения на синтагмы")

    durations_stat_json = "data/stats/male_alloph_durations.json"
    durations_stats = get_object_from_json(durations_stat_json)

    lev_distances = []
    probabilities = []
    for j, variant in enumerate(txt_cluster_distribution_variants):
        lev = 0
        sum_probability = 0
        for i, part in enumerate(variant):
            ac_dur = ac_synt_durations[i]
            probable_duration_arr = [0, 0]

            for x in text_syntagmas[j][i]:
                if x not in tuple("0.,!?;:-"):
                    if x.startswith(tuple("aeoiuy")):
                        x = x[0] + '0'
                    probable_duration_arr[0] += durations_stats[x][0]
                    probable_duration_arr[1] += durations_stats[x][1]

            if txt_cluster_distribution_variants.index(variant) == 60:
                print("")

            # Вычисляем вероятность P(ac_dur - epsilon < N < ac_dur + epsilon)
            mean = probable_duration_arr[0]
            std_dev = probable_duration_arr[1]
            probability = stats.norm.cdf(ac_dur + 0.2 * ac_dur, loc=mean, scale=std_dev) - stats.norm.cdf(
                ac_dur - 0.2 * ac_dur, loc=mean, scale=std_dev)

            sum_probability += probability
            if probability < 0.1:
                continue

            lev += make_str_alignment(part, ac_syntagmas[i])
        probabilities.append(round(sum_probability / len(variant), 2))
        if lev == 0: lev = 0.01
        lev_distances.append(lev)
        num_var = round((j + 1) / len(txt_cluster_distribution_variants), 0)
        if num_var != 0 and num_var % 10 == 0:
            print(f"Найдена вероятность {j + 1}/{len(txt_cluster_distribution_variants)} варианта")

    lev_distances_norm = [round(max(lev_distances), 2) / x for x in lev_distances]
    probable_levenstein = [round(x * probabilities[l], 4) for l, x in enumerate(lev_distances_norm)]
    best_var = list(probable_levenstein).index(max(probable_levenstein))

    best_syntagma_distribution = syntagma_distribution_variants[best_var]

    return best_syntagma_distribution


def get_cluster_index_for_border(text_clusters, index):
    cluster_ind = 0
    index_cl = 0
    for cl1, cl2 in zip(text_clusters, text_clusters[1:]):
        if cl1[0] <= index < cl2[0]:
            cluster_ind = index_cl
        index_cl += 1
    return cluster_ind


def make_most_probable_syntagma_distribution_2(ac_string: str, txt_clusters, word_bound_indexes, ac_synt_durations,
                                               txt_arr, reverse=False):
    #TODO: доделать вычисление границ наизнанку
    # считать акустическую длину по транскрипции от начала сигнала, а не от конца прошлого отрезка.
    # Вычислять границы с меньшим левенштейном для каждого участка, потом находить пересечения.
    # Если пересечений нет - последний наиболее вероятный из левого края (т.к. меньше расхождение по акустике)
    if reverse:
        ac_string = ac_string[::-1]
        txt_arr = txt_arr[::-1]
        ac_synt_durations = ac_synt_durations[::-1]
        num_symb = len(txt_arr)
        word_bound_indexes_new = [num_symb - i for i in word_bound_indexes]
        word_bound_indexes_new.append(0)
        word_bound_indexes = sorted(word_bound_indexes_new)[:-1]
        txt_clusters_new = [(0, txt_clusters[-1][1])]
        for cl1, cl2 in zip(txt_clusters[::-1], txt_clusters[::-1][1:]):
            txt_clusters_new.append((num_symb - cl1[0], cl2[1]))
        txt_clusters = txt_clusters_new

    ac_syntagmas = [i for i in ac_string.split("4") if i]

    durations_stat_json = "data/stats/male_alloph_durations.json"
    durations_stats = get_object_from_json(durations_stat_json)

    synt_bord_indexes = [0]
    starting_point = 0
    best_operations_all = []

    for i, synt in enumerate(ac_syntagmas):
        probable_synt_indexes = []
        probabilities = []
        synt_duration = ac_synt_durations[i]
        for index in word_bound_indexes:
            txt_part = txt_arr[starting_point: index]
            txt_part_updated = [ch if not ch.startswith(tuple("aeoiuy")) else ch + "0" for ch in txt_part]
            mean_len_part = sum(durations_stats[ch][0] for ch in txt_part_updated if ch in durations_stats.keys())
            std_len_part = sum(durations_stats[ch][1] for ch in txt_part_updated if ch in durations_stats.keys())
            if mean_len_part + std_len_part < synt_duration:
                continue
            if mean_len_part - std_len_part > synt_duration:
                break
            probable_synt_indexes.append(index)
            probability = stats.norm.cdf(synt_duration + 0.2 * synt_duration, loc=mean_len_part, scale=std_len_part) - stats.norm.cdf(
                synt_duration - 0.2 * synt_duration, loc=mean_len_part, scale=std_len_part)
            probabilities.append(probability)

        lev_dist_for_probable_distributions = []
        operations_list = []
        probabilities_for_selected = []
        for num, index in enumerate(probable_synt_indexes):
            part = txt_clusters[get_cluster_index_for_border(txt_clusters, starting_point): get_cluster_index_for_border(txt_clusters, index) + 1]

            part_distribution = delete_same_symbols("".join([str(x[1]) for x in part]))
            (lev, operations) = make_str_alignment(part_distribution, ac_syntagmas[i], 0, [])
            lev_dist_for_probable_distributions.append(lev)
            operations_list.append(operations)
            probabilities_for_selected.append(probabilities[num])

        min_lev = min(lev_dist_for_probable_distributions)
        best_vars_by_lev = [ind for ind, variant in enumerate(lev_dist_for_probable_distributions) if variant == min_lev]
        best_var = probable_synt_indexes[best_vars_by_lev[-1]]
        best_operation = operations_list[best_vars_by_lev[-1]]
        synt_bord_indexes.append(best_var)
        starting_point = best_var + 1
        best_operations_all.append(best_operation)

    best_synt_distribution = []
    for ind1, ind2 in zip(synt_bord_indexes, synt_bord_indexes[1:]):
        best_synt_distribution.append([ind1, ind2])

    best_synt_distribution[-1][1] = len(txt_arr) - 1

    return best_synt_distribution, best_operations_all
