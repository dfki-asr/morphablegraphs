import argparse
from morphablegraphs.utilities.db_interface import align_motions_in_db
from utils import get_session

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run alignment.')
    parser.add_argument('db_url', nargs='?', help='db_url')
    parser.add_argument('skeleton_name', nargs='?', help='skeleton_name')
    parser.add_argument('c_id', nargs='?', help='collection id')
    parser.add_argument( "--user", nargs='+',default=None, help='User for server access')
    parser.add_argument( "--password", nargs='+',default=None, help='Password for server access')
    parser.add_argument( "--token", nargs='+',default=None, help='Encrypted user password for server access')
  
    args = parser.parse_args()
    if args.db_url is not None and args.skeleton_name is not None and args.c_id is not None:
       session = get_session(args)
       align_motions_in_db(args.db_url, args.skeleton_name, args.c_id, session=session)
    else:
        print("not enough args")

