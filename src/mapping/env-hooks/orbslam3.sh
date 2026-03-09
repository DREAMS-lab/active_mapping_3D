_ORBSLAM3_ROOT="$HOME/workspaces/zed_ws/ORB_SLAM3"
_PANGOLIN_ROOT="$HOME/workspaces/zed_ws/Pangolin"

_ORBSLAM3_PATHS="$_ORBSLAM3_ROOT/lib:$_ORBSLAM3_ROOT/Thirdparty/DBoW2/lib:$_ORBSLAM3_ROOT/Thirdparty/g2o/lib:$_PANGOLIN_ROOT/build/src"

case ":${LD_LIBRARY_PATH}:" in
  *":$_ORBSLAM3_ROOT/lib:"*) ;;
  *) export LD_LIBRARY_PATH="$_ORBSLAM3_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

unset _ORBSLAM3_ROOT _PANGOLIN_ROOT _ORBSLAM3_PATHS
