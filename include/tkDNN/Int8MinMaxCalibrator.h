#ifndef INT8MINMAXCALIBRATOR_H
#define INT8MINMAXCALIBRATOR_H

#include <vector>
#include <assert.h>
#include <algorithm>
#include <iterator>
#include <stdint.h>
#include <iostream>
#include <string>
#include "NvInfer.h"

#include <fstream>
#include <iomanip>

#include "Int8BatchStream.h"

#include "tkdnn.h"
#include "utils.h"

/*
 * Int8EntropyCalibrator implements the INT8 calibrator to achieve the
 * INT8 quantization. It uses a BatchStream stream to scroll through 
 * images data. It also implements the calibration cache, a way to 
 * save the calibration process results to reduce the running time: 
 * the calibration process takes a long time.
 */
class Int8MinMaxCalibrator : public nvinfer1::IInt8MinMaxCalibrator {
public:
	Int8MinMaxCalibrator(BatchStream& stream, int firstBatch, const std::string& calibTableFilePath, 
							const std::string& inputBlobName, bool readCache = true);
	virtual ~Int8MinMaxCalibrator() { checkCuda(cudaFree(mDeviceInput)); }
	int getBatchSize() const NOEXCEPT override { return mStream.getBatchSize(); }
	bool getBatch(void* bindings[], const char* names[], int nbBindings) NOEXCEPT override;
	const void* readCalibrationCache(size_t& length) NOEXCEPT override;
	void writeCalibrationCache(const void* cache, size_t length) NOEXCEPT override;

private:
	BatchStream mStream;
	const std::string mCalibTableFilePath{ nullptr };
	const std::string mInputBlobName;
	bool mReadCache{ true };

	size_t mInputCount;
	void* mDeviceInput{ nullptr };
	std::vector<char> mCalibrationCache;
};

#endif //INT8CALIBRATORMINMAX_H
