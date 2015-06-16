#include <ecto/ecto.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd/rgbd.hpp>

using ecto::tendrils;

struct RescaledRegisteredDepth {
  static void declare_params(tendrils& params) {
  }

  static void declare_io(const tendrils& params, tendrils& in, tendrils& out) {
    in.declare(&RescaledRegisteredDepth::depth_in_, "depth", "The depth image to rescale.").required(true);
    in.declare(&RescaledRegisteredDepth::mask_in_, "mask", "A mask of the size of depth, that will be resized too.");
    in.declare(&RescaledRegisteredDepth::image_in_, "image", "The rgb image.").required(true);
    in.declare(&RescaledRegisteredDepth::K_in_, "K", "The camera intrinsics matrix of the depth camera to rescale.");

    out.declare(&RescaledRegisteredDepth::depth_out_, "depth", "The rescaled depth image.");
    out.declare(&RescaledRegisteredDepth::mask_out_, "mask", "The rescaled depth image.");
    out.declare(&RescaledRegisteredDepth::K_out_, "K", "The rescaled camera intrinsics matrix of the depth camera.");
  }

  int process(const tendrils& in, const tendrils& out) {
    cv::Size dsize = depth_in_->size(), isize = image_in_->size();
    K_in_->copyTo(*K_out_);

    if (dsize == isize) {
      rescaleDepth(*depth_in_, CV_32F, *depth_out_);
      *mask_out_ = *mask_in_;
      return ecto::OK;
    }

    cv::Mat depth;
    cv::Mat valid_mask;
    rescaleDepth(*depth_in_, CV_32F, depth);

    float factor = float(isize.height) / dsize.height;
    cv::Mat output(isize, depth.type());
    //resize into the subregion of the correct aspect ratio
    int cols_nbr = dsize.width  *factor;
    int col_start = round((isize.width - cols_nbr)/2);
    cv::Mat subregion(output.colRange(col_start, cols_nbr+col_start));
    //use nearest neighbor to prevent discontinuities causing bogus depth.
    cv::resize(depth, subregion, subregion.size(), CV_INTER_NN);
    if (col_start > 0)
    output.colRange(0, col_start-1).setTo(
        cv::Scalar(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN()));
    output.colRange(cols_nbr+col_start, output.cols).setTo(
        cv::Scalar(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN()));
    *depth_out_ = output;

    //rescale the calibration matrix
    K_out_->at<float>(0, 0) = K_out_->at<float>(0, 0) * static_cast<float>(isize.width) / static_cast<float>(dsize.width); //K_out_->at<float>(0, 0) * factor;
    K_out_->at<float>(1, 1) = K_out_->at<float>(1, 1) * static_cast<float>(isize.height) / static_cast<float>(dsize.height);
    K_out_->at<float>(0, 2) = static_cast<float>(isize.width-1) / 2.0f;
    K_out_->at<float>(1, 2) = static_cast<float>(isize.height-1) / 2.0f;

    if (!mask_in_->empty()) {
      assert(mask_in_->size() == depth_in_->size());
      cv::Mat mask(isize, CV_8U);
      cv::Mat subregion(mask.colRange(col_start, cols_nbr+col_start));
      //use nearest neighbor to prevent discontinuities causing bogus depth.
      cv::resize(*mask_in_, subregion, subregion.size(), CV_INTER_NN);
      if (col_start > 0)
        mask.colRange(0, col_start-1).setTo(cv::Scalar(0, 0, 0));
      mask.colRange(cols_nbr+col_start, output.cols).setTo(cv::Scalar(0, 0, 0));
      *mask_out_ = mask;
    }

    return ecto::OK;
  }
  ecto::spore<cv::Mat> image_in_, depth_in_, depth_out_, mask_in_, mask_out_, K_in_, K_out_;
};

ECTO_CELL(base, RescaledRegisteredDepth, "RescaledRegisteredDepth",
    "Rescale the openni depth image to be the same size as the image if necessary.")
