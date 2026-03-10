#pragma once

#include "typedef.h"

class CudaImageProcess
{
public:

	CudaImageProcess(void);
	~CudaImageProcess(void);

	bool Initialize(void);
	bool ImageBlur(void);
	bool IncreaseResolution(void);
	bool StoreImage(void);
	void Close(void);


private:

	bool applyKawase(uint32_t applyTimes);

private:

	Buf_s* h_rBuf;
	Buf_s* h_gBuf;
	Buf_s* h_bBuf;
	uint32_t width, height;

	uint8_t* d_originalR;
	uint8_t* d_originalG;
	uint8_t* d_originalB;

	uint8_t* d_rBuf;
	uint8_t* d_gBuf;
	uint8_t* d_bBuf;
	size_t pitch;

};