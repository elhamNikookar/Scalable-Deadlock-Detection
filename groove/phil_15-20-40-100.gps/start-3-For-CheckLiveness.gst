<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<gxl xmlns="http://www.gupro.de/GXL/gxl-1.0.dtd">
    <graph role="graph" edgeids="false" edgemode="directed" id="start-3-For-CheckLiveness">
        <attr name="transitionLabel">
            <string></string>
        </attr>
        <attr name="enabled">
            <string>true</string>
        </attr>
        <attr name="priority">
            <string>0</string>
        </attr>
        <attr name="printFormat">
            <string></string>
        </attr>
        <attr name="remark">
            <string></string>
        </attr>
        <attr name="$version">
            <string>curly</string>
        </attr>
        <node id="n1">
            <attr name="layout">
                <string>237 239 32 24</string>
            </attr>
        </node>
        <node id="n0">
            <attr name="layout">
                <string>358 252 38 48</string>
            </attr>
        </node>
        <node id="n239">
            <attr name="layout">
                <string>104 200 38 48</string>
            </attr>
        </node>
        <node id="n237">
            <attr name="layout">
                <string>217 187 32 24</string>
            </attr>
        </node>
        <node id="n233">
            <attr name="layout">
                <string>278 180 32 24</string>
            </attr>
        </node>
        <node id="n240">
            <attr name="layout">
                <string>242 41 55 72</string>
            </attr>
        </node>
        <edge from="n1" to="n1">
            <attr name="label">
                <string>Fork</string>
            </attr>
        </edge>
        <edge from="n0" to="n0">
            <attr name="label">
                <string>Phil</string>
            </attr>
        </edge>
        <edge from="n0" to="n0">
            <attr name="label">
                <string>think</string>
            </attr>
        </edge>
        <edge from="n0" to="n1">
            <attr name="label">
                <string>right</string>
            </attr>
        </edge>
        <edge from="n0" to="n233">
            <attr name="label">
                <string>left</string>
            </attr>
        </edge>
        <edge from="n239" to="n239">
            <attr name="label">
                <string>Phil</string>
            </attr>
        </edge>
        <edge from="n239" to="n239">
            <attr name="label">
                <string>think</string>
            </attr>
        </edge>
        <edge from="n239" to="n237">
            <attr name="label">
                <string>right</string>
            </attr>
        </edge>
        <edge from="n239" to="n1">
            <attr name="label">
                <string>left</string>
            </attr>
        </edge>
        <edge from="n237" to="n237">
            <attr name="label">
                <string>Fork</string>
            </attr>
        </edge>
        <edge from="n233" to="n233">
            <attr name="label">
                <string>Fork</string>
            </attr>
        </edge>
        <edge from="n240" to="n240">
            <attr name="label">
                <string>Phil</string>
            </attr>
        </edge>
        <edge from="n240" to="n240">
            <attr name="label">
                <string>hungry</string>
            </attr>
        </edge>
        <edge from="n240" to="n240">
            <attr name="label">
                <string>specific</string>
            </attr>
        </edge>
        <edge from="n240" to="n237">
            <attr name="label">
                <string>left</string>
            </attr>
        </edge>
        <edge from="n240" to="n233">
            <attr name="label">
                <string>right</string>
            </attr>
        </edge>
    </graph>
</gxl>
