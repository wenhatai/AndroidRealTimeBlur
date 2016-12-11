LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#OPENCV_LIB_TYPE:=STATIC
include sdk/native/jni/OpenCV.mk

LOCAL_MODULE    := blur
LOCAL_SRC_FILES := jni_part.cpp
LOCAL_LDLIBS +=  -llog -ldl

include $(BUILD_SHARED_LIBRARY)
