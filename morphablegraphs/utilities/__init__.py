try:
    from mgrd import Skeleton as MGRDSkeleton
    from mgrd import SkeletonNode as MGRDSkeletonNode
    has_mgrd = True
except ImportError:
    has_mgrd = False
    pass


def convert_to_mgrd_skeleton(skeleton):
    if not has_mgrd:
        return None
    def create_mgrd_node(mg_node, parent):
        mgrd_node = MGRDSkeletonNode(mg_node.node_name, parent, mg_node.offset, mg_node.rotation)
        mgrd_node.fixed = mg_node.fixed
        return mgrd_node

    def populate(skeleton, mgrd_node):
        node = skeleton.nodes[mgrd_node.name]
        for child in node.children:
            child_node = create_mgrd_node(child, mgrd_node)
            mgrd_node.add_child(child_node)
            populate(skeleton, child_node)
    root_node = create_mgrd_node(skeleton.nodes[skeleton.root], None)
    populate(skeleton, root_node)
    return MGRDSkeleton(root_node)
