ic_examples = {
'''
# the person is jumping. bring right knee to chest at the highest point of the jump
def right_knee_to_chest():
    # load the motion that needs to be edited
    load_motion("motion_0")

    # the original motion is that the person is jumping. the desired edit is bring right knee to chest.
    # the joints involved are the right knee and the right hip.
    identified_joints = ["right_knee", "right_hip"]
   
    # the motion timing is relative to another joint: when the waist is highest. The desired edit is a verb

    # bend right knee
    do_rotate("right_knee", "flex", time=when_joint("waist", "highest", is_verb=True))
    # flex the right hip to bring the knee higher
    do_rotate("right_hip", "flex", time=when_joint("waist", "highest", is_verb=True))

    # save edited motion
    save_motion("motion_1")
''',
'''
# the person is standing still. leap into the air at the start of the motion
def jump():
    # the original motion is that the person is standing still. The desired edit is leap into the air.
    # load the motion that needs to be edited
    load_motion("motion_0")

    # the primary joint involved is the entire body, summarized by the waist.
    identified_joints = ["waist"]
    # the motion timing is relative to a point in the global motion: the start of the motion. The desired edit is a verb.

    # the waist represents the whole body, and we move it upwards to simulate jumping
    do_translate("waist", "up", time=at_global_moment("start_of_motion", is_verb=True))
    
    # save edited motion
    save_motion("motion_1")
''',
'''
# the person is dancing. get lower to the ground the entire time
def squat():
    # load motion
    load_motion("motion_0")

    # the original motion is that the person is dancing. The desired edit is get lower to the ground.
    # the primary joint involved is the entire body, summarized by the waist.
    identified_joints = ["waist"]
    # the motion timing is relative to points in the global motion: the entire motion. The desired edit is not a verb.
    # the waist needs to be closer to the ground
    do_translate("waist", "down", time=at_global_moment("entire_motion", is_verb=False))

    # save edited motion
    save_motion("motion_1")
'''
}

ANIMATION_FEWSHOTS = [
    (
        "Bring right knee to chest during jump",
        [
            "Instructions: Bring right knee to chest during jump\nCurrent Animation: The person is jumping"
        ],
        [
            "Let me analyze this edit:\n1. The original motion is a jumping motion\n2. We need to modify the right knee and hip to bring knee to chest\n3. This should happen when the waist is at its highest point during the jump\n4. We'll need to flex both the knee and hip joints"
        ],
        [
            '''# load motion that needs to be edited
load_motion("motion_0")

# the original motion is that the person is jumping. the desired edit is bring right knee to chest.
# the joints involved are the right knee and the right hip.
identified_joints = ["right_knee", "right_hip"]
   
# the motion timing is relative to another joint: when the waist is highest. The desired edit is a verb

# bend right knee
do_rotate("right_knee", "flex", time=when_joint("waist", "highest", is_verb=True))
# flex the right hip to bring the knee higher
do_rotate("right_hip", "flex", time=when_joint("waist", "highest", is_verb=True))

# save edited motion
save_motion("motion_1")'''
        ]
    ),
    (
        "Make the standing person leap into the air at the start",
        [
            "Instructions: Make the standing person leap into the air at the start\nCurrent Animation: The person is standing still"
        ],
        [
            "Let me analyze this edit:\n1. The original motion is standing still\n2. We need to create an upward leap\n3. This affects the whole body, which we can control via the waist\n4. The leap should happen at the start of the motion"
        ],
        [
            '''# load the motion that needs to be edited
load_motion("motion_0")

# the original motion is that the person is standing still. The desired edit is leap into the air.
# the primary joint involved is the entire body, summarized by the waist.
identified_joints = ["waist"]
# the motion timing is relative to a point in the global motion: the start of the motion. The desired edit is a verb.

# the waist represents the whole body, and we move it upwards to simulate jumping
do_translate("waist", "up", time=at_global_moment("start_of_motion", is_verb=True))

# save edited motion
save_motion("motion_1")'''
        ]
    ),
    (
        "Lower the dancing person's position throughout the motion",
        [
            "Instructions: Lower the dancing person's position throughout the motion\nCurrent Animation: The person is dancing"
        ],
        [
            "Let me analyze this edit:\n1. The original motion is a dance\n2. We need to lower the entire body position\n3. This affects the whole body, controlled via the waist\n4. The lowering should happen throughout the entire motion"
        ],
        [
            '''# load motion
load_motion("motion_0")

# the original motion is that the person is dancing. The desired edit is get lower to the ground.
# the primary joint involved is the entire body, summarized by the waist.
identified_joints = ["waist"]
# the motion timing is relative to points in the global motion: the entire motion. The desired edit is not a verb.
# the waist needs to be closer to the ground
do_translate("waist", "down", time=at_global_moment("entire_motion", is_verb=False))

# save edited motion
save_motion("motion_1")'''
        ]
    )
]
