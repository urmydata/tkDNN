#ifndef INT8BATCHSTREAM_H
#define INT8BATCHSTREAM_H

#include <vector>
#include <assert.h>
#include <algorithm>
#include <iterator>
#include <stdint.h>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <signal.h>
#include <stdlib.h>    
#include <unistd.h>
#include <mutex>

#include "NvInfer.h"
#include "utils.h"
#include "tkdnn.h"

/*
 * BatchStream implements the stream for the INT8 calibrator. 
 * It reads the two files .txt with the list of image file names 
 * and the list of label file names. 
 * It then iterates on images and labels.
 */
class BatchStream {
public:
	BatchStream(tk::dnn::dataDim_t dim, int batchSize, int maxBatches, const std::string& fileimglist);
	virtual ~BatchStream() { }
	void reset(int firstBatch);
	bool next();
	void skip(int skipCount);
	float *getBatch() { return mBatch.data(); }
	int getBatchesRead() const { return mBatchCount; }
	int getBatchSize() const { return mBatchSize; }
	nvinfer1::DimsNCHW getDims() const { return mDims; }
	float* getFileBatch() { return &mFileBatch[0]; }
	void readInListFile(const std::string& dataFilePath, std::vector<std::string>& mListIn);
	void readCVimage(std::string inputFileName, std::vector<float>& res, bool fixshape = true);
	bool update();

private:
	int mBatchSize{ 0 };
	int mMaxBatches{ 0 };
	int mBatchCount{ 0 };
	int mFileCount{ 0 };
	int mFileBatchPos{ 0 };
	int mImageSize{ 0 };

	nvinfer1::DimsNCHW mDims;
	std::vector<float> mBatch;
	std::vector<float> mFileBatch;

	int mHeight;
	int mWidth;
	std::string mFileImgList;
	std::vector<std::string> mListImg;
};

#endif //INT8BATCHSTREAM
