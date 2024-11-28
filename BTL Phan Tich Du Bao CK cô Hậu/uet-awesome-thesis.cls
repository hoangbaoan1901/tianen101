% -------------------------------------------------------------------
%  @LaTeX-class-file{
%     filename        = "hust-thesis.cls",
%     version         = "0.0.1",
%     date            = "2021",
%     codetable       = "ISO/ASCII",
%     keywords        = "LaTeX, hust-thesis",
%     supported       = "send email to ngocjr7@gmail.com",
%     docstring       = "A thesis class for a HUST dissertation. This is an adaptation based on Harvard's template."
% --------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uet-awesome-thesis}[2021/12/28 University of Engineering and Technology Thesis Class]

\newcommand{\mylanguage}{en}

\DeclareOption{en}{\renewcommand{\mylanguage}{en}}
\DeclareOption{vi}{\renewcommand{\mylanguage}{vi}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClass[oneside]{book}
\usepackage{scrextend}
\changefontsizes{13pt}

% Required packages
\RequirePackage{graphicx}
\RequirePackage{hyperref}
\RequirePackage[toc]{glossaries}
\RequirePackage{glossary-mcols}
\RequirePackage{here}
\RequirePackage[toc]{appendix}

\RequirePackage{amssymb}
\RequirePackage{amsmath}
\RequirePackage{mathtools}
\RequirePackage{mathspec} %loads fontspec as well
\RequirePackage[ruled,vlined]{algorithm2e}

\RequirePackage[numbers, comma, sort&compress]{natbib}
% titlesec
\RequirePackage{titlesec}
\RequirePackage[tight,nice]{units}
\RequirePackage{verbatim}
\RequirePackage[palatino]{quotchap}
\RequirePackage{ragged2e}
% colors
\RequirePackage{color}
\definecolor{Crimson}{rgb}{0.6471, 0.1098, 0.1882}

\hypersetup{
    linktocpage,
    colorlinks,
    citecolor=blue,
    filecolor=black,
    linkcolor=black,
    urlcolor=blue,
}

\RequirePackage{url}
\RequirePackage[titles]{tocloft}

\setcounter{tocdepth}{3}
\setcounter{secnumdepth}{3}

% \renewcommand{\cftchapfont}{\normalsize \scshape}
\setlength{\parindent}{1cm}
\setlength{\parskip}{6pt}

\RequirePackage{pdfsync} %do pdf synchronization [andy]
\RequirePackage[
    type={CC},
    modifier={by-nc-sa},
    version={3.0},
]{doclicense}

% \usepackage[closeFloats, noSeparatorLine]{fltpage} %use the custom modified fltpage package
% \RequirePackage{afterpage} 

\synctex=1 % turn synctex on automatically to sync between pdf viewer [andy]

%   list an entire bibliography entry inline. Useful for acknowledging when my paper was previously published
\RequirePackage{bibentry}
\nobibliography*

% \RequirePackage{lettrine} % big letter at start of chapter
\RequirePackage[a4paper]{geometry}
\geometry{left=3cm, right=2cm, top=2.5cm, bottom=3cm}

\RequirePackage{fancyhdr}
\pagestyle{plain} % options: empty , plain , fancy

% \RequirePackage[palatino]{quotchap} % chaper number 
% \definecolor{chaptergrey}{rgb}{0.6,0,0}

\RequirePackage{afterpage}


\RequirePackage{titling}
\RequirePackage{setspace}
\RequirePackage{booktabs} % for much better looking tables
\RequirePackage[labelfont={bf},justification=RaggedRight,margin=0pt, figurewithin=section, tablewithin=section]{caption}

% line alignment
\linespread{1.08333333}
\addtolength{\footnotesep}{\footnotesep}
\justifying
%\raggedright


% TITLE FORMAT 1
% \titleformat{\chapter}[display]
%     {\Huge}
%     {\chaptertitlename\ \thechapter}{20pt}{\Huge\sloppy}

% \titleformat{\section}
%     {\Large}
%     {\thesection}{1em}{\Large}

% \titleformat{\subsection}
%     {\large}
%     {\thesubsection}{1em}{\large}

% \titleformat{\subsubsection}[runin]
%     {\bf}
%     {\thesubsection}{1em}{}

% \titleformat{\paragraph}[runin]
%     {\it}
%     {}{1em}{}

% TITLE FORMAT 2 (Default)
\titleformat{\chapter}[display]
{\normalfont\huge\bfseries}{\chaptertitlename\ \thechapter}{20pt}{\Huge}
\titleformat{\section}[block]
{\normalfont\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}
{\normalfont\large\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}
{\normalfont\normalsize\bfseries}{\thesubsubsection}{1em}{}
\titleformat{\paragraph}[runin]
{\normalfont\normalsize\bfseries}{\theparagraph}{1em}{}
\titleformat{\subparagraph}[runin]
{\normalfont\normalsize\bfseries}{\thesubparagraph}{1em}{}
\titlespacing*{\chapter}     {0pt}{50pt}{40pt}
\titlespacing*{\section}     {0pt}{3.5ex plus 1ex minus .2ex}{2.3ex plus .2ex}
\titlespacing*{\subsection}   {0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}
\titlespacing*{\subsubsection}{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}
\titlespacing*{\paragraph}   {0pt}{3.25ex plus 1ex minus .2ex}{1em}
\titlespacing*{\subparagraph} {\parindent}{3.25ex plus 1ex minus .2ex}{1em}

% some definitions
\def\degreeyear#1{\gdef\@degreeyear{#1}}
\def\degreemonth#1{\gdef\@degreemonth{#1}}
\def\advisor#1{\gdef\@advisor{#1}}
\def\university#1{\gdef\@university{#1}}
\def\universitycity#1{\gdef\@universitycity{#1}}
\def\major#1{\gdef\@major{#1}}


\newcommand{\isenglish}[3][\mylanguage]{
    \ifthenelse{\equal{#1}{en}}{#2}{#3}
}

\isenglish{
    \RequirePackage[english]{babel}
}{
    \RequirePackage[vietnamese]{babel}
}

\RequirePackage{tikz}
\usetikzlibrary{calc}
\usetikzlibrary{decorations.pathmorphing}


% remove hyperlink, mostly used for citation in caption and headline
\newcommand{\nohyper}[1]{\protect\NoHyper#1\protect\endNoHyper}

\renewcommand\bibname{References}
\renewcommand\listfigurename{Listing of Figures}

% undo the wrong changes made by mathspec
\makeatletter
\let\RequirePackage\original@RequirePackage
\let\usepackage\RequirePackage
\makeatother