
def combine_pgt_gt(pgt_file, gt_file, save_path):
    with open(save_path, 'w+') as f:
        with open(pgt_file, 'r') as fa:
            f.write(fa.read())
        with open(gt_file, 'r') as fa:
            f.write(fa.read())

