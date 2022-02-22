import argparse
from anim_utils.retargeting.analytical import Retargeting, generate_joint_map
from anim_utils.utilities.db_interface import authenticate, get_motion_list_from_remote_db, load_skeleton_from_db, retarget_motion_in_db
from utils import get_session



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run retargeting.')
    parser.add_argument('db_url', nargs='?', help='db_url')
    parser.add_argument('src_skeleton', nargs='?', help='skeleton_name')
    parser.add_argument('dest_skeleton', nargs='?', help='skeleton_name')
    parser.add_argument('c_id', nargs='?', help='collection id')
    parser.add_argument('is_aligned', nargs='?',default=0, help='int')
    parser.add_argument('src_scale', nargs='?',default=1.0, help='float')
    parser.add_argument('place_on_ground', nargs='?',default=1, help='int')
    parser.add_argument( "--user", nargs='+',default=None, help='User for server access')
    parser.add_argument( "--password", nargs='+',default=None, help='Password for server access')
    parser.add_argument( "--token", nargs='+',default=None, help='Encrypted user password for server access')
  
    args = parser.parse_args()
    if args.db_url is not None and args.src_skeleton is not None and args.dest_skeleton is not None and args.c_id is not None:
        session = get_session(args)
        src_skeleton = load_skeleton_from_db(args.db_url, args.src_skeleton)
        dest_skeleton = load_skeleton_from_db(args.db_url, args.dest_skeleton)
        joint_map = generate_joint_map(src_skeleton.skeleton_model, dest_skeleton.skeleton_model)
        retargeting = Retargeting(src_skeleton, dest_skeleton, joint_map, float(args.src_scale), additional_rotation_map=None, place_on_ground=bool(args.place_on_ground))
        for motion in get_motion_list_from_remote_db(args.db_url, args.c_id, args.src_skeleton, args.is_aligned):
            retarget_motion_in_db(args.db_url, retargeting, motion[0], motion[1], args.c_id, args.dest_skeleton, session=session)
    else:
        print("not enough args")
