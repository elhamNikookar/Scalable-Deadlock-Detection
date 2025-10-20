<?xml version="1.0" encoding="UTF-8"?>
<gxl xmlns="http://www.gupro.de/GXL/gxl-1.0.dtd">
  <graph id="s1" edgeids="false" edgemode="undirected">
    <attr name="state">
      <string>s1</string>
    </attr>
    <node id="phil_0">
      <attr name="type">
        <string>Philosopher</string>
      </attr>
      <attr name="state">
        <string>think</string>
      </attr>
      <attr name="left_fork">
        <string>fork_0</string>
      </attr>
      <attr name="right_fork">
        <string>fork_1</string>
      </attr>
    </node>
    <node id="phil_1">
      <attr name="type">
        <string>Philosopher</string>
      </attr>
      <attr name="state">
        <string>think</string>
      </attr>
      <attr name="left_fork">
        <string>fork_1</string>
      </attr>
      <attr name="right_fork">
        <string>fork_2</string>
      </attr>
    </node>
    <node id="phil_2">
      <attr name="type">
        <string>Philosopher</string>
      </attr>
      <attr name="state">
        <string>think</string>
      </attr>
      <attr name="left_fork">
        <string>fork_2</string>
      </attr>
      <attr name="right_fork">
        <string>fork_3</string>
      </attr>
    </node>
    <node id="phil_3">
      <attr name="type">
        <string>Philosopher</string>
      </attr>
      <attr name="state">
        <string>think</string>
      </attr>
      <attr name="left_fork">
        <string>fork_3</string>
      </attr>
      <attr name="right_fork">
        <string>fork_4</string>
      </attr>
    </node>
    <node id="phil_4">
      <attr name="type">
        <string>Philosopher</string>
      </attr>
      <attr name="state">
        <string>think</string>
      </attr>
      <attr name="left_fork">
        <string>fork_4</string>
      </attr>
      <attr name="right_fork">
        <string>fork_5</string>
      </attr>
    </node>
    <node id="phil_5">
      <attr name="type">
        <string>Philosopher</string>
      </attr>
      <attr name="state">
        <string>think</string>
      </attr>
      <attr name="left_fork">
        <string>fork_5</string>
      </attr>
      <attr name="right_fork">
        <string>fork_0</string>
      </attr>
    </node>
    <node id="fork_0">
      <attr name="type">
        <string>Fork</string>
      </attr>
      <attr name="state">
        <string>available</string>
      </attr>
    </node>
    <node id="fork_1">
      <attr name="type">
        <string>Fork</string>
      </attr>
      <attr name="state">
        <string>available</string>
      </attr>
    </node>
    <node id="fork_2">
      <attr name="type">
        <string>Fork</string>
      </attr>
      <attr name="state">
        <string>available</string>
      </attr>
    </node>
    <node id="fork_3">
      <attr name="type">
        <string>Fork</string>
      </attr>
      <attr name="state">
        <string>available</string>
      </attr>
    </node>
    <node id="fork_4">
      <attr name="type">
        <string>Fork</string>
      </attr>
      <attr name="state">
        <string>available</string>
      </attr>
    </node>
    <node id="fork_5">
      <attr name="type">
        <string>Fork</string>
      </attr>
      <attr name="state">
        <string>available</string>
      </attr>
    </node>
    <edge from="phil_0" to="fork_0">
      <attr name="label">
        <string>left</string>
      </attr>
    </edge>
    <edge from="phil_0" to="fork_1">
      <attr name="label">
        <string>right</string>
      </attr>
    </edge>
    <edge from="phil_1" to="fork_1">
      <attr name="label">
        <string>left</string>
      </attr>
    </edge>
    <edge from="phil_1" to="fork_2">
      <attr name="label">
        <string>right</string>
      </attr>
    </edge>
    <edge from="phil_2" to="fork_2">
      <attr name="label">
        <string>left</string>
      </attr>
    </edge>
    <edge from="phil_2" to="fork_3">
      <attr name="label">
        <string>right</string>
      </attr>
    </edge>
    <edge from="phil_3" to="fork_3">
      <attr name="label">
        <string>left</string>
      </attr>
    </edge>
    <edge from="phil_3" to="fork_4">
      <attr name="label">
        <string>right</string>
      </attr>
    </edge>
    <edge from="phil_4" to="fork_4">
      <attr name="label">
        <string>left</string>
      </attr>
    </edge>
    <edge from="phil_4" to="fork_5">
      <attr name="label">
        <string>right</string>
      </attr>
    </edge>
    <edge from="phil_5" to="fork_5">
      <attr name="label">
        <string>left</string>
      </attr>
    </edge>
    <edge from="phil_5" to="fork_0">
      <attr name="label">
        <string>right</string>
      </attr>
    </edge>
  </graph>
</gxl>