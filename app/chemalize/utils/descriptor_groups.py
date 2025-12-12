"""
Descriptor Groups Parser
Parses Descriptors_group.txt file to extract descriptor group information
"""
import os
from typing import Dict, List, Tuple


def parse_descriptor_groups(file_path: str) -> Dict[str, Dict[str, any]]:
    """
    Parse the Descriptors_group.txt file to extract descriptor groups.

    Args:
        file_path: Path to Descriptors_group.txt file

    Returns:
        Dictionary with group IDs as keys and group info as values:
        {
            '1': {
                'id': '1',
                'name': 'Constitutional indices',
                'full_name': '1. Constitutional indices',
                'descriptors': ['MW', 'AMW', 'Sv', ...]
            },
            ...
        }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Descriptor groups file not found: {file_path}")

    groups = {}
    current_group_id = None
    current_group_name = None
    current_descriptors = []
    in_descriptor_list = False

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Check if this is a group header (e.g., "1. Constitutional indices")
            if line and line[0].isdigit() and '. ' in line:
                # Save previous group if exists
                if current_group_id is not None:
                    groups[current_group_id] = {
                        'id': current_group_id,
                        'name': current_group_name,
                        'full_name': f"{current_group_id}. {current_group_name}",
                        'descriptors': current_descriptors.copy()
                    }

                # Parse new group
                parts = line.split('. ', 1)
                current_group_id = parts[0]
                current_group_name = parts[1] if len(parts) > 1 else ''
                current_descriptors = []
                in_descriptor_list = False

            elif line == 'Name':
                # Start of descriptor list
                in_descriptor_list = True

            elif line == '':
                # Empty line - might end descriptor list
                continue

            elif in_descriptor_list and line and current_group_id is not None:
                # This is a descriptor name
                current_descriptors.append(line)

        # Save last group
        if current_group_id is not None:
            groups[current_group_id] = {
                'id': current_group_id,
                'name': current_group_name,
                'full_name': f"{current_group_id}. {current_group_name}",
                'descriptors': current_descriptors.copy()
            }

    return groups


def get_descriptors_for_groups(groups_dict: Dict[str, Dict], selected_group_ids: List[str]) -> List[str]:
    """
    Get all descriptor names for selected groups.

    Args:
        groups_dict: Dictionary from parse_descriptor_groups()
        selected_group_ids: List of group IDs to include (e.g., ['1', '3', '5'])

    Returns:
        List of descriptor names from all selected groups
    """
    descriptors = []
    for group_id in selected_group_ids:
        if group_id in groups_dict:
            descriptors.extend(groups_dict[group_id]['descriptors'])
    return descriptors


def filter_dataframe_by_groups(df, groups_dict: Dict[str, Dict], selected_group_ids: List[str],
                                keep_non_descriptors: bool = True) -> Tuple[any, List[str]]:
    """
    Filter dataframe to include only descriptors from selected groups.

    Args:
        df: Pandas DataFrame
        groups_dict: Dictionary from parse_descriptor_groups()
        selected_group_ids: List of group IDs to include
        keep_non_descriptors: If True, keep columns that are not in any descriptor group

    Returns:
        Tuple of (filtered_df, list_of_kept_descriptor_columns)
    """
    import pandas as pd

    # Get all descriptors from selected groups
    selected_descriptors = get_descriptors_for_groups(groups_dict, selected_group_ids)

    # Get all possible descriptors (from all groups) to identify non-descriptor columns
    all_descriptors = []
    for group_info in groups_dict.values():
        all_descriptors.extend(group_info['descriptors'])

    all_descriptors_set = set(all_descriptors)

    # Determine which columns to keep
    columns_to_keep = []
    descriptor_columns_kept = []

    for col in df.columns:
        if col in selected_descriptors:
            # This column is in our selected groups
            columns_to_keep.append(col)
            descriptor_columns_kept.append(col)
        elif keep_non_descriptors and col not in all_descriptors_set:
            # Only keep if it's a non-numeric column (identifier like Sample ID, Name, etc.)
            # This ensures numeric descriptors NOT in groups file are excluded
            if not pd.api.types.is_numeric_dtype(df[col]):
                columns_to_keep.append(col)

    filtered_df = df[columns_to_keep].copy()

    return filtered_df, descriptor_columns_kept


def get_group_summary(groups_dict: Dict[str, Dict], selected_group_ids: List[str]) -> Dict[str, any]:
    """
    Get summary information about selected groups.

    Args:
        groups_dict: Dictionary from parse_descriptor_groups()
        selected_group_ids: List of group IDs

    Returns:
        Dictionary with summary information
    """
    total_descriptors = 0
    group_names = []

    for group_id in selected_group_ids:
        if group_id in groups_dict:
            group_info = groups_dict[group_id]
            total_descriptors += len(group_info['descriptors'])
            group_names.append(group_info['name'])

    return {
        'num_groups': len(selected_group_ids),
        'total_descriptors': total_descriptors,
        'group_names': group_names,
        'group_ids': selected_group_ids
    }
