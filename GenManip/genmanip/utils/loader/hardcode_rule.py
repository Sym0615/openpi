"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""


def verify_cup_and_plate(
    object_config_key_list: list[str],
    key: str,
    key_index: int,
    replaced_uid: str,
    added_uid_list: list[str],
) -> bool | None:
    key_index = object_config_key_list.index(key)
    if key == "cup1":
        if (
            "plate1" not in object_config_key_list[:key_index]
            and "plate2" not in object_config_key_list[:key_index]
        ):
            return True
        elif (
            "plate2" in object_config_key_list[:key_index]
            and "plate1" not in object_config_key_list[:key_index]
        ):
            plate_index = object_config_key_list.index("plate2")
            if replaced_uid.replace("cup_", "") != added_uid_list[plate_index].replace(
                "plate_", ""
            ):
                return True
            else:
                return False
        else:
            plate_index = object_config_key_list.index("plate1")
            if replaced_uid.replace("cup_", "") != added_uid_list[plate_index].replace(
                "plate_", ""
            ):
                return False
            else:
                return True
    elif key == "cup2":
        if (
            "plate2" not in object_config_key_list[:key_index]
            and "plate1" not in object_config_key_list[:key_index]
        ):
            return True
        elif (
            "plate1" in object_config_key_list[:key_index]
            and "plate2" not in object_config_key_list[:key_index]
        ):
            plate_index = object_config_key_list.index("plate1")
            if replaced_uid.replace("cup_", "") != added_uid_list[plate_index].replace(
                "plate_", ""
            ):
                return True
            else:
                return False
        else:
            plate_index = object_config_key_list.index("plate2")
            if replaced_uid.replace("cup_", "") != added_uid_list[plate_index].replace(
                "plate_", ""
            ):
                return False
            else:
                return True
    elif key == "plate1":
        if (
            "cup1" not in object_config_key_list[:key_index]
            and "cup2" not in object_config_key_list[:key_index]
        ):
            return True
        elif (
            "cup1" not in object_config_key_list[:key_index]
            and "cup2" in object_config_key_list[:key_index]
        ):
            cup_index = object_config_key_list.index("cup2")
            if replaced_uid.replace("plate_", "") != added_uid_list[cup_index].replace(
                "cup_", ""
            ):
                return True
            else:
                return False
        else:
            cup_index = object_config_key_list.index("cup1")
            if replaced_uid.replace("plate_", "") != added_uid_list[cup_index].replace(
                "cup_", ""
            ):
                return False
            else:
                return True
    elif key == "plate2":
        if "cup2" not in object_config_key_list[:key_index]:
            return True
        elif (
            "cup1" in object_config_key_list[:key_index]
            and "cup2" not in object_config_key_list[:key_index]
        ):
            cup_index = object_config_key_list.index("cup1")
            if replaced_uid.replace("plate_", "") != added_uid_list[cup_index].replace(
                "cup_", ""
            ):
                return True
            else:
                return False
        else:
            cup_index = object_config_key_list.index("cup2")
            if replaced_uid.replace("plate_", "") != added_uid_list[cup_index].replace(
                "cup_", ""
            ):
                return False
            else:
                return True
