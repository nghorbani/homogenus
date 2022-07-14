import homogenus
from homogenus import *
from homogenus.tf import homogenus_infer

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-tm", "--trained_model_dir", default="./homogenus/trained_models/tf/", help="The path to the directory holding homogenus trained models in TF.")
    parser.add_argument("-ii", "--images_indir", required= True, help="Directory of the input images.")
    parser.add_argument("-oi", "--openpose_indir", required=True, help="Directory of openpose keypoints, e.g. json files.")
    parser.add_argument("-io", "--images_outdir", default=None, help="Directory to put predicted gender overlays. If not given, wont produce any overlays.")
    parser.add_argument("-oo", "--openpose_outdir", default=None, help="Directory to put the openpose gendered keypoints. If not given, it will augment the original openpose json files.")

    ps = parser.parse_args()

    hg = homogenus.tf.homogenus_infer.Homogenus_infer(trained_model_dir=ps.trained_model_dir)
    hg.predict_genders(images_indir=ps.images_indir, openpose_indir=ps.openpose_indir,
                       images_outdir=ps.images_outdir, openpose_outdir=ps.openpose_outdir)



main()