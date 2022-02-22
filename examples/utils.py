
from anim_utils.utilities.db_interface import authenticate

def get_session(args):
   session = None
   if args.user is not None and args.password is not None:
       result = authenticate(args.db_url, args.user, args.password)
       if "token" in result:
            session = {"user": args.user, "token": result["token"]}
   elif args.user is not None and args.token is not None:
       session = {"user": args.user, "token": args.token}
   return session
