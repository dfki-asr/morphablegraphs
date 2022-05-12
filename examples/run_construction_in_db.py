import argparse
from morphablegraphs.utilities.db_interface import create_motion_model_in_db, get_standard_config
from utils import get_session


if __name__ == "__main__":
    config = get_standard_config()
    parser = argparse.ArgumentParser(description='Run modeling.')
    parser.add_argument('db_url', nargs='?', help='db_url')
    parser.add_argument('skeleton_name', nargs='?', help='skeleton_name')
    parser.add_argument('c_id', nargs='?', help='collection id')
    parser.add_argument('c_name', nargs='?', help='collection name')
    parser.add_argument('basis_factor', nargs='?', default=config["n_spatial_basis_factor"], help='spline_basis_factor')
    parser.add_argument( "--joint_filter", nargs='+',default=None, help='Sequence of joint names to model')
    parser.add_argument( "--user", nargs='+',default=None, help='User for server access')
    parser.add_argument( "--password", nargs='+',default=None, help='Password for server access')
    parser.add_argument( "--token", nargs='+',default=None, help='Encrypted user password for server access')
    args = parser.parse_args()
    if args.db_url is not None and args.skeleton_name is not None and args.c_id is not None and args.c_name is not None:
       session = get_session(args)
       create_motion_model_in_db(args.db_url, 
                                    args.skeleton_name,
                                    args.c_id, 
                                    args.c_name, 
                                    float(args.basis_factor), args.joint_filter, session=session)
    else:
        print("Not enough arguments")

