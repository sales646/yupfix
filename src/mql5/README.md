# MQL5 Setup Instructions

To run the `ZmqServer.mq5` Expert Advisor, you need to set up the ZeroMQ library for MetaTrader 5.

## Prerequisites

1.  **MetaTrader 5**: Installed and running.
2.  **ZeroMQ DLL**: `libzmq.dll` (usually v4.x).
3.  **MQL-ZMQ Wrapper**: An include file (`Zmq.mqh`) to interface with the DLL.

## Installation Steps

1.  **Copy Files**:
    *   Copy `ZmqServer.mq5` to your MT5 Data Folder: `MQL5/Experts/Antigravity/`.
    *   Copy `libzmq.dll` to `MQL5/Libraries/`.
    *   Copy `Zmq.mqh` (and dependencies) to `MQL5/Include/Zmq/`.

2.  **Enable DLLs**:
    *   In MT5, go to **Tools > Options > Expert Advisors**.
    *   Check **"Allow WebRequest for listed URL"**.
    *   Check **"Allow DLL imports"**.

3.  **Run**:
    *   Open a chart (e.g., EURUSD).
    *   Drag `ZmqServer` onto the chart.
    *   Check the "Experts" tab in the Toolbox for "ZMQ Server Started".

## Dependencies

This code assumes you are using a standard MQL-ZMQ binding. If you don't have one, we recommend:
*   [Dingmaotu/mql-zmq](https://github.com/dingmaotu/mql-zmq)
