from genmanip.utils.standalone.transform_utils import (
    adjust_orientation,
    adjust_translation_along_quaternion,
)


def prepare_grasp_motion_planning_payload(init_grasp, steps=30, padding=0.08):
    init_grasp["orientation"] = adjust_orientation(init_grasp["orientation"])
    action_list = []
    action_list.append(
        {
            "name": "pre_grasp",
            "translation": adjust_translation_along_quaternion(
                init_grasp["translation"], init_grasp["orientation"], padding + 0.06
            ),
            "orientation": init_grasp["orientation"],
            "steps": steps,
            "grasp": False,
        }
    )
    action_list.append(
        {
            "name": "grasp",
            "translation": adjust_translation_along_quaternion(
                init_grasp["translation"], init_grasp["orientation"], padding
            ),
            "orientation": init_grasp["orientation"],
            "steps": steps,
            "grasp": False,
        }
    )
    action_list.append(
        {
            "name": "post_grasp",
            "translation": adjust_translation_along_quaternion(
                init_grasp["translation"], init_grasp["orientation"], padding + 0.22
            ),
            "orientation": init_grasp["orientation"],
            "steps": steps,
            "grasp": True,
        }
    )
    return action_list
