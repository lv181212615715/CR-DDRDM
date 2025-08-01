from pathlib import Path
import pandas as pd
import numpy as np
import os


def final_integration(data, sorted_objects_clusters, similarity_matrix):
    """Integrate sorted object clusters into a final consensus ranking"""

    def custom_sort(row1, row2):
        """Custom sorting function comparing two rows"""
        # Ensure index alignment
        row1, row2 = row1.align(row2, join='inner', axis=0)
        diff = (row1 - row2)
        return diff.sum()

    def intercluster_sorting(dfs):
        """Sort clusters based on their first row comparison"""
        # Extract first row from each dataframe
        first_rows = [df.iloc[0, :-1] for df in dfs]
        first_rows_df = pd.DataFrame(first_rows).reset_index(drop=True)

        # Sort clusters based on comparison with first cluster
        sorted_indices = sorted(range(len(first_rows_df)),
                                key=lambda i: -custom_sort(first_rows_df.iloc[i], first_rows_df.iloc[0]))

        # Return sorted list of dataframes
        return [dfs[i] for i in sorted_indices]

    def merge_sorted_segments(segment1, segment2):
        """Merge two sorted segments while maintaining order"""
        merged_segment = []
        i, j = 0, 0

        # Merge segments using custom sort comparison
        while i < len(segment1) and j < len(segment2):
            if custom_sort(segment1.iloc[i], segment2.iloc[j]) >= 0:
                merged_segment.append(segment1.iloc[i])
                i += 1
            else:
                merged_segment.append(segment2.iloc[j])
                j += 1

        # Append remaining elements
        merged_segment.extend(segment1.iloc[i:].values.tolist())
        merged_segment.extend(segment2.iloc[j:].values.tolist())

        return pd.DataFrame(merged_segment)

    def merge_and_sort_segments(result, df, similarity_matrix, merged_size):
        """Merge new dataframe into result using similarity matrix"""
        similarity_matrix = pd.DataFrame(similarity_matrix)

        def get_sorted_indices(similarity_array):
            """Get indices sorted by similarity (descending)"""
            return np.argsort(similarity_array)[::-1]

        # Case 1: Simple concatenation if properly ordered
        if custom_sort(result.iloc[-1], df.iloc[0]) > 0:
            return pd.concat([result, df]), merged_size

        # Case 2: Partial overlap in ordering
        elif custom_sort(result.iloc[-1], df.iloc[-1]) > 0:
            # Find best matching segments using similarity matrix
            similarity_matrix_df_to_result = similarity_matrix.loc[df.index[0], result.index].values
            x1_indices = get_sorted_indices(similarity_matrix_df_to_result)

            similarity_matrix_result_to_df = similarity_matrix.loc[result.index[-1], df.index].values
            x2_indices = get_sorted_indices(similarity_matrix_result_to_df)

            # Merge overlapping segments
            segment1 = result.iloc[x1_indices[0]:]
            segment2 = df.iloc[:x2_indices[0] + 1]
            merged_segment = merge_sorted_segments(segment1, segment2)

            # Reconstruct result with merged segment
            new_result = pd.concat([result.iloc[:x1_indices[0]], merged_segment, df.iloc[x2_indices[0] + 1:]])
            return new_result, merged_size + merged_segment.shape[0]

        # Case 3: Complete reordering needed
        elif custom_sort(result.iloc[-1], df.iloc[-1]) < 0:
            # Find all possible matching segments
            similarity_matrix_df_to_result1 = similarity_matrix.loc[df.index[0], result.index].values
            x1_indices = get_sorted_indices(similarity_matrix_df_to_result1)

            similarity_matrix_df_to_result2 = similarity_matrix.loc[df.index[-1], result.index].values
            x2_indices = get_sorted_indices(similarity_matrix_df_to_result2)

            # Try all valid segment combinations
            for x1_idx in x1_indices:
                for x2_idx in x2_indices:
                    if x1_idx <= x2_idx:
                        segment1 = result.iloc[x1_idx:x2_idx + 1]
                        segment2 = df
                        merged_segment = merge_sorted_segments(segment1, segment2)

                        new_result = pd.concat([result.iloc[:x1_idx], merged_segment, result.iloc[x2_idx + 1:]])
                        return new_result, merged_size + merged_segment.shape[0]

        # Case 4: Equal end points
        elif custom_sort(result.iloc[-1], df.iloc[-1]) == 0:
            similarity_matrix_df_to_result = similarity_matrix.loc[df.index[0], result.index].values
            x1_indices = get_sorted_indices(similarity_matrix_df_to_result)

            # Merge entire dataframe with matching segment
            segment1 = result.iloc[x1_indices[0]:]
            segment2 = df
            merged_segment = merge_sorted_segments(segment1, segment2)

            new_result = pd.concat([result.iloc[:x1_indices[0]], merged_segment])
            return new_result, merged_size + merged_segment.shape[0]

        return result, merged_size

    def merge_sorted_dataframes(dfs):
        """Merge all sorted dataframes into final result"""
        if not dfs:
            return pd.DataFrame(), 0

        result = dfs[0]
        merged_size = 0

        for df in dfs[1:]:
            if df.empty:
                continue
            result, merged_size = merge_and_sort_segments(result, df, similarity_matrix, merged_size)

        return result, merged_size

    def re_sort(df):
        """Sort dataframe subgroups based on custom sort function"""
        sorted_subsets = []
        unique_sorts = df['sort'].unique()

        for sort_val in unique_sorts:
            subset = df[df['sort'] == sort_val]
            sorted_indices = sorted(range(len(subset)),
                                    key=lambda i: custom_sort(subset.iloc[i, :-1], subset.iloc[0, :-1]),
                                    reverse=True)
            sorted_subsets.append(subset.iloc[sorted_indices])

        return pd.concat(sorted_subsets)

    # Prepare dataframes for merging
    dfs = []
    for cluster in sorted_objects_clusters:
        merged = pd.concat([data, cluster.iloc[:, 0]], axis=1, join='inner')
        merged.rename(columns={1: 'sort'}, inplace=True)
        merged.sort_values('sort', ascending=True, inplace=True)
        dfs.append(re_sort(merged))

    # Perform inter-cluster sorting and final merging
    dfs = intercluster_sorting(dfs)
    final_result, merged_size = merge_sorted_dataframes(dfs)

    return pd.DataFrame(final_result.index), merged_size