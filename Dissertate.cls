% -------------------------------------------------------------------
%  @LaTeX-class-file{
%     filename        = "Dissertate.cls",
%     version         = "2.0",
%     date            = "25 March 2014",
%     codetable       = "ISO/ASCII",
%     keywords        = "LaTeX, Dissertate",
%     supported       = "Send email to suchow@post.harvard.edu.",
%     docstring       = "Class for a dissertation."
% --------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Dissertate}[2014/03/25 v2.0 Dissertate Class]
%\LoadClass[10pt, oneside, a5paper]{book}
%\documentclass[9pt]{extarticle}

\LoadClass[10pt,twoside]{extbook}%a5paper

\usepackage{setspace}
\usepackage{amsmath}
\usepackage[no-math]{fontspec}
\usepackage{soul}
%\usepackage{xcolor}
\usepackage[cmyk]{xcolor}
\usepackage{amssymb}
\usepackage{braket}
\usepackage{physics}
\usepackage{bbold}
\usepackage{verbatim}
\usepackage{sansmath}
\usepackage{siunitx}
\usepackage{wrapfig}
\usepackage{bm}
%\usepackage{placeins}

\usepackage[version=3]{mhchem} % Formula subscripts using \ce{}
%\usepackage{chemformula} % Formula subscripts using \ch{}

\usepackage{tikz}
\usetikzlibrary{calc,arrows,arrows.meta,decorations.pathmorphing,intersections}
%\usepackage[font={small,sf},labelfont={bf},labelsep=endash]{caption}

\usepackage[section]{placeins}
\usepackage[euler]{textgreek}
\usepackage[mathletters]{ucs}
\usepackage[T2A,T1]{fontenc}       % Use modern font encodings
\usepackage[utf8]{inputenc}
\usepackage[russian,english]{babel}


%\usepackage[T1]{fontenc} % Use modern font encodings

\definecolor{ForestGreens}{HTML}{228B22}
\definecolor{MyYellow}{HTML}{FFC133}



%##################################################
%###########	FOR PYTHON     ####################
\usepackage{listings}

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}
\definecolor{coderednumber}{RGB}{222,37,26}

\lstdefinestyle{mystyle}{
	backgroundcolor=\color{backcolour},   
	commentstyle=\color{codegreen},
	keywordstyle=\color{magenta},
	numberstyle=\tiny\color{codegray},
	stringstyle=\color{codepurple},
	basicstyle=\ttfamily\footnotesize,
	breakatwhitespace=false,         
	breaklines=true,                 
	captionpos=b,                    
	keepspaces=true,                 
	numbers=left,                    
	numbersep=5pt,                  
	showspaces=false,                
	showstringspaces=false,
	showtabs=false,                  
	tabsize=2
}

\lstset{style=mystyle,xleftmargin=0.08\textwidth,xrightmargin=0.05\textwidth}
%##################################################


%\usepackage[utf8]{inputenc}
%\usepackage[T2A,T1]{fontenc}
\usepackage{fontspec}
\setmainfont[
Ligatures=TeX,
Extension=.otf,
BoldFont=cmunbx,
ItalicFont=cmunti,
BoldItalicFont=cmunbi,
]{cmunrm}
\setsansfont[
Ligatures=TeX,
Extension=.otf,
BoldFont=cmunsx,
ItalicFont=cmunsi,
]{cmunss}

\usepackage[russian,english]{babel}
\usepackage{hyphenat}


\RequirePackage[
papersize={170mm,240mm},
%a4paper,layoutsize={170mm,240mm},layoutoffset={20mm,28.5mm},
textwidth=125mm,
%textwidth=128mm,
vmargin={30mm,30mm},
outer=18mm,
%outer=25mm,
%showcrop,showframe,
]{geometry}



%\usepackage{polyglossia}
%\usepackage{polyglossia}
%\setmainlanguage{english}
%\setotherlanguage{russian} 


%\DeclareUnicodeCharacter{03BD}{\greektextnu} 
%\usepackage[Latin,Greek]{ucharclasses}
%\newfontfamily{\greekfont}{GFSDidot}[
%Extension=.otf,
%UprightFont=*,
%ItalicFont=*Italic,
%BoldFont=*Bold,
%BoldItalicFont=*BoldItalic,
%Scale=MatchLowercase,
%]

%\usepackage{needspace}
%\usepackage{mdframed}
%\mdfdefinestyle{MyFrame}{%
%	linecolor=black,
%	outerlinewidth=2pt,
%	roundcorner=0pt,
%	innertopmargin=\baselineskip,
%	innerbottommargin=\baselineskip,
%	innerrightmargin=30pt,
%	innerleftmargin=10pt}
%	backgroundcolor=gray!50!white

\makeatletter
\def\mathcolor#1#{\@mathcolor{#1}}
\def\@mathcolor#1#2#3{%
	\protect\leavevmode
	\begingroup\color#1{#2}#3\endgroup
}
%\makeatother

\usepackage{tikz}
\usetikzlibrary{calc,arrows,arrows.meta,decorations.pathmorphing,intersections}
\definecolor{ForestGreens}{HTML}{228B22}


\newcommand*\circled[1]{\tikz[baseline=(char.base)]{
		\node[shape=circle,draw,inner sep=2pt] (char) {#1};}}
	
\newcommand{\fp}{{\it FACPACK}}

%\usepackage[utf8]{inputenc}
%\usepackage[greek]{babel}

%\RequirePackage[superscript,biblabel,nomove]{cite}
\RequirePackage[superscript,move]{cite}
%\bibliographystyle{naturemag}

%\RequirePackage{pdfsync} %do pdf synchronization [andy]
\RequirePackage[closeFloats, noSeparatorLine]{packages/fltpage2} 
%use the custom modified fltpage package
\synctex=1 % turn synctex on automatically to sync between pdf viewer

% Referencesintro.
%\renewcommand\bibname{Referencesintro}
%\bibliography{referencesintro}
%\RequirePackage[super,comma,numbers]{natbib}
%\renewcommand{\bibnumfmt}[1]{[#1]}
%\RequirePackage[palatino]{quotchap}
%\renewcommand*{\chapterheadstartvskip}{\vspace*{-0.5\baselineskip}}
%\renewcommand*{\chapterheadendvskip}{\vspace{1.3\baselineskip}}

% Settings for the author block
\RequirePackage{authblk}
\setlength{\affilsep}{1.5em}
%\renewcommand\Authfont{\fontsize{12}{12}\usefont{OT1}{phv}{b}{n}}% changed 23-11-2017
%\renewcommand\Affilfont{\fontsize{10}{12}\usefont{OT1}{phv}{m}{n}}% changed 23-11-2017

%\makeatletter
%\DeclareRobustCommand\citenum
%{\begingroup
%	%      \NAT@swatrue\let\NAT@ctype\z@\NAT@parfalse\let\textsuperscript\NAT@spacechar% DELETED
%	\NAT@swatrue\let\NAT@ctype\z@\NAT@parfalse\let\textsuperscript\relax% NEW
%	\NAT@citexnum[][]}
%\makeatother



\def\shrinkage{2.1mu}
\def\vecsign{\mathchar"017E}
\def\dvecsign{\smash{\stackon[-1.95pt]{\mkern-\shrinkage\vecsign}{\rotatebox{180}{$\mkern-\shrinkage\vecsign$}}}}
\def\dvec#1{\def\useanchorwidth{T}\stackon[-4.2pt]{#1}{\,\dvecsign}}
\usepackage{stackengine}
\stackMath
\usepackage{graphicx}




%\newcommand\shrinkage{2.1mu}
%\newcommand\vecsign{\mathchar"017E}
%\newcommand\dvecsign{\smash{\stackon[-1.5pt]{\vecsign}{\rotatebox{180}{$\vecsign$}}}}
%\newcommand\dvec[1]{\def\useanchorwidth{T}\stackon[-3.5pt]{#1}{\,\dvecsign}}
%\usepackage{stackengine}
%\stackMath
%\usepackage{graphicx}

\renewcommand\Authfont{\fontsize{10}{10}\usefont{OT1}{phv}{b}{n}}
\renewcommand\Affilfont{\fontsize{8}{10}\usefont{OT1}{phv}{m}{n}}
	
%\RequirePackage[left=2cm,%
%                right=2cm,%
%                top=2.25cm,%
%                bottom=2.25cm,%
%                headheight=12pt,%
%                letterpaper]{geometry}%
                
\RequirePackage[labelfont={bf,rm},%
                labelsep=period]{caption}%justification=raggedright

\RequirePackage[colorlinks=true, allcolors=blue]{hyperref}

% article meta data
%
\newcommand{\keywords}[1]{\def\@keywords{#1}}

\def\xabstract{abstract}
\long\def\abstract#1\end#2{\def\two{#2}\ifx\two\xabstract 
\long\gdef\theabstract{\ignorespaces#1}
\def\go{\end{abstract}}\else
\typeout{^^J^^J PLEASE DO NOT USE ANY \string\begin\space \string\end^^J
COMMANDS WITHIN ABSTRACT^^J^^J}#1\end{#2}
\gdef\theabstract{\vskip12pt BADLY FORMED ABSTRACT: PLEASE DO
NOT USE {\tt\string\begin...\string\end} COMMANDS WITHIN
THE ABSTRACT\vskip12pt}\let\go\relax\fi
\go}



%\usepackage{graphicx}

%\usepackage{mathtools}
%\DeclarePairedDelimiter\abs{\lvert}{\rvert}%
%\DeclarePairedDelimiter\norm{\lVert}{\rVert}%
%
%% Swap the definition of \abs* and \norm*, so that \abs
%% and \norm resizes the size of the brackets, and the 
%% starred version does not.
%\makeatletter
%\let\oldabs\abs
%\def\abs{\@ifstar{\oldabs}{\oldabs*}}
%%
%\let\oldnorm\norm
%\def\norm{\@ifstar{\oldnorm}{\oldnorm*}}
%\makeatother


\usepackage[symbol*]{footmisc}
%\usepackage[perpage]{footmisc}
% Using symbols for footnotes
\renewcommand{\thefootnote}{\fnsymbol{footnote}}

\usepackage{hyperref}

\newenvironment{widetext}{}{}

%\usepackage{breqn}

% Van Harvard (origineel):


\RequirePackage{etoolbox}

% Line spacing: dsingle/ddouble
%   Whether to use single- or doublespacing.
\newtoggle{DissertateSingleSpace}
\togglefalse{DissertateSingleSpace}
\DeclareOption{dsingle}{
    \toggletrue{DissertateSingleSpace}
    \ClassWarning{Dissertate}{Single-spaced mode on.}
}
%\DeclareOption{ddouble}{\togglefalse{DissertateSingleSpace}}

\ProcessOptions\relax

% Line Spacing
%   Define two line spacings: one for the body, and one that is more compressed.
%\iftoggle{DissertateSingleSpace}{
%    \newcommand{\dnormalspacing}{1.2}
%    \newcommand{\dcompressedspacing}{1.0}
%}{
%    \newcommand{\dnormalspacing}{2.0}
%    \newcommand{\dcompressedspacing}{1.2}
%}


%###############    SPACING ####################
%################################################
\iftoggle{DissertateSingleSpace}{
	\newcommand{\dnormalspacing}{1.}
	\newcommand{\dcompressedspacing}{1.0}
	%\newcommand{\dcompressedspacing}{0.8}	
}{
	\newcommand{\dnormalspacing}{1.2}
	\newcommand{\dcompressedspacing}{1.0}
	%\newcommand{\dcompressedspacing}{0.8}	
}
%################################################
%################################################


	
% Block quote with compressed spacing
\let\oldquote\quote
\let\endoldquote\endquote
\renewenvironment{quote}
    {\begin{spacing}{\dcompressedspacing}\oldquote}
    {\endoldquote\end{spacing}}

% Itemize with compressed spacing
%\let\olditemize\itemize
%\let\endolditemize\enditemize
%\renewenvironment{itemize}
%    {\begin{spacing}{\dcompressedspacing}\olditemize}
%    {\endolditemize\end{spacing}}

% Enumerate with compressed spacing % changed 23-11-2017
\let\oldenumerate\enumerate
\let\endoldenumerate\endenumerate
\renewenvironment{enumerate}
    {\begin{spacing}{\dcompressedspacing}\oldenumerate}
    {\endoldenumerate\end{spacing}}

% Text layout.

% \RequirePackage[width=5.75in, letterpaper]{geometry} % changed 23-11-2017

%\usepackage[a5paper,bindingoffset=0.5cm,%
%left=1cm,right=2cm,top=2cm,bottom=2cm,%
%footskip=0.5cm]{geometry}

\usepackage{ragged2e}
%\RaggedRight
\RequirePackage{graphicx}
\usepackage{fixltx2e}
%\parindent 12pt
\parindent 10pt % changed 23-11-2017
\RequirePackage{lettrine}
\RequirePackage{setspace}
\RequirePackage{verbatim}
\usepackage{roboto}

% Fonts.
%\RequirePackage{color}
%\RequirePackage[cmyk]{xcolor}
%\usepackage[cmyk]{xcolor}
\usepackage{hyperref}
\RequirePackage{url}
\RequirePackage{amssymb}
\RequirePackage{mathspec}
\RequirePackage{mathrsfs}
%\setmathsfont(Digits,Latin,Greek)[Numbers={Proportional}]{EB Garamond}  
%\newfontfamily\greekfont[Script=Greek]{EB Garamond}% changed 23-11-2017: orig; EB Garamond % Can't we get it to do: "Garamond Italic"?
%\setmathrm{Roboto Slab Regular}  % changed 23-11-2017 from EB Garamond
%\setmathrm{Roboto Slab Regular}%Commented out on 17-06-2018
%\setmathbf{EB Garamond}
\AtBeginEnvironment{tabular}{\addfontfeature{RawFeature=+tnum}}
\widowpenalty=300
\clubpenalty=300
%\setromanfont[Numbers=OldStyle, Ligatures={Common, TeX}, Scale=1.0]{EB Garamond} % is ook Lato geweest


%%%%%%%%%%      FONT      %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%\setromanfont[Numbers=OldStyle, Ligatures={Common, TeX}, Scale=1,BoldFont={Roboto Slab Bold},ItalicFont={Roboto Italic},BoldItalicFont={Roboto Bold Italic}]{Roboto Slab Regular}%Lato
%\newfontfamily{\smallcaps}[RawFeature={+c2sc,+scmp}]{Roboto Slab Thin}
%\setsansfont[Scale=MatchLowercase, BoldFont={Roboto Slab Bold}]{Roboto Slab Regular}
%\setmonofont[Scale=MatchLowercase]{Source Code Pro}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\newlength{\figwidth}\setlength{\figwidth}{115mm}
\newlength{\smallfigwidth}\setlength{\smallfigwidth}{80mm}
\setkeys{Gin}{width=\figwidth}



%%%%%%      FIGURE CAPTIONS      %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[labelfont={bf,rm,small,singlespacing},
                textfont={rm,small,singlespacing},
                justification={justified},%,RaggedRight
                singlelinecheck=false,
%                margin=0.5cm,
				width=\figwidth,
                figurewithin=chapter,
                tablewithin=chapter]{caption}
                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
                
                
\renewcommand{\thefootnote}{\fnsymbol{footnote}}

% Headings and headers.
\RequirePackage{fancyhdr}
\RequirePackage[tiny, md, sc]{titlesec}
\setlength{\headheight}{20pt}
\pagestyle{plain}
\RequirePackage{titling}



%\titleformat{\section}
%{\color{cyan}\itshape\LARGE}
%{\llap{\thesection\ }}
%{0pt}


%\titleformat{\section}
%{\normalfont\Large\bfseries}{\thesection}{1em}{}

%%%%%% SECTION & SUBSECTION FORMAT %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%\newcommand{\sectionbreak}{\clearpage}

\titleformat{\section}
{\normalfont\Large\bfseries}{\thesection}{0.9em}{}
%{\normalfont\Large\bfseries\uppercase}{\thesection}{1em}{}
\titleformat{\subsection}
{\normalfont\large\bfseries}{\thesubsection}{0.9em}{}
\titleformat{\subsubsection}
{\normalfont\normalsize\bfseries}{\thesubsubsection}{0.9em}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Front matter.
\setcounter{tocdepth}{2}
\usepackage[titles]{tocloft}
\usepackage[titletoc,title,toc]{appendix}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}
\renewcommand{\cftchapfont}{\normalsize \scshape}
\renewcommand\listfigurename{Listing of figures}
\renewcommand\listtablename{Listing of tables}

% Endmatter
\renewcommand{\setthesection}{\arabic{chapter}.A\arabic{section}}

% References.
\renewcommand\bibname{References}
\RequirePackage[super,comma,numbers,sort&compress]{natbib}
%\renewcommand{\bibnumfmt}[1]{[#1]}
\RequirePackage[palatino]{quotchap}
\renewcommand*{\chapterheadstartvskip}{\vspace*{-0.5\baselineskip}}
\renewcommand*{\chapterheadendvskip}{\vspace{1.3\baselineskip}}

% An environment for paragraph-style section.
\providecommand\newthought[1]{%
   \addvspace{1.0\baselineskip plus 0.5ex minus 0.2ex}%
   \noindent\textsc{#1}%
}

% Align reference numbers so that they do not cause an indent.
\newlength\mybibindent
\setlength\mybibindent{0pt}
\renewenvironment{thebibliography}[1]
    {\chapter*{\bibname}%
     \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}%
     \list{\@biblabel{\@arabic\c@enumiv}}
          {\settowidth\labelwidth{\@biblabel{999}}
           \leftmargin\labelwidth
            \advance\leftmargin\dimexpr\labelsep+\mybibindent\relax\itemindent-\mybibindent
           \@openbib@code
           \usecounter{enumiv}
           \let\p@enumiv\@empty
           \renewcommand\theenumiv{\@arabic\c@enumiv}}
     \sloppy
     \clubpenalty4000
     \@clubpenalty \clubpenalty
     \widowpenalty4000%
     \sfcode`\.\@m}
    {\def\@noitemerr
      {\@latex@warning{Empty `thebibliography' environment}}
     \endlist}
 
 
 %% Two-clumn stuff:
%\makeatletter
% \renewenvironment{thebibliography}[1]{%
% 	\section*{\refname \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}}%
% 	\begin{multicols}{2}
% 		\list{\@biblabel{\@arabic\c@enumiv}}{%
% 			\settowidth\labelwidth{\@biblabel{#1}}%
% 			\leftmargin\labelwidth
% 			\advance\leftmargin\labelsep
% 			\@openbib@code
% 			\usecounter{enumiv}%
% 			\let\p@enumiv\@empty
% 			\renewcommand\theenumiv{\@arabic\c@enumiv}}%
% 		\sloppy
% 		\clubpenalty4000
% 		\@clubpenalty \clubpenalty
% 		\widowpenalty4000%
% 		\sfcode`\.\@m}
% 	{
% 		\def\@noitemerr{\@latex@warning{Empty `thebibliography' environment}}%
% 		\endlist\end{multicols}
% }
%\makeatother
 
%\usepackage{multicol}
%\usepackage{etoolbox}
%\patchcmd{\thebibliography}{\section*{\refname}}
%{\begin{multicols}{2}[\section*{\refname}]}{}{}
% 	\patchcmd{\endthebibliography}{\endlist}{\endlist\end{multicols}}{}{}

% Some definitions.
\def\advisor#1{\gdef\@advisor{#1}}
\def\coadvisorOne#1{\gdef\@coadvisorOne{#1}}
\def\coadvisorTwo#1{\gdef\@coadvisorTwo{#1}}
\def\committeeInternal#1{\gdef\@committeeInternal{#1}}
\def\committeeInternalOne#1{\gdef\@committeeInternalOne{#1}}
\def\committeeInternalTwo#1{\gdef\@committeeInternalTwo{#1}}
\def\committeeExternal#1{\gdef\@committeeExternal{#1}}
\def\degreeyear#1{\gdef\@degreeyear{#1}}
\def\degreemonth#1{\gdef\@degreemonth{#1}}
\def\degreeterm#1{\gdef\@degreeterm{#1}}
\def\degree#1{\gdef\@degree{#1}}
\def\department#1{\gdef\@department{#1}}
\def\field#1{\gdef\@field{#1}}
\def\university#1{\gdef\@university{#1}}
\def\universitycity#1{\gdef\@universitycity{#1}}
\def\universitystate#1{\gdef\@universitystate{#1}}
\def\programname#1{\gdef\@programname{#1}}
\def\pdOneName#1{\gdef\@pdOneName{#1}}
\def\pdOneSchool#1{\gdef\@pdOneSchool{#1}}
\def\pdOneYear#1{\gdef\@pdOneYear{#1}}
\def\pdTwoName#1{\gdef\@pdTwoName{#1}}
\def\pdTwoSchool#1{\gdef\@pdTwoSchool{#1}}
\def\pdTwoYear#1{\gdef\@pdTwoYear{#1}}
% School name and location
\university{University of Amsterdam}
\universitycity{Amsterdam}
\universitystate{The Netherlands}

% School color found from university's graphic identity site:
% http://isites.harvard.edu/icb/icb.do?keyword=k75408&pageid=icb.page392732

%\definecolor{SchoolColor}{rgb}{0.6471, 0.1098, 0.1882} % Crimson
%\definecolor{chaptergrey}{rgb}{0.6471, 0.1098, 0.1882} % for chapter numbers


%\definecolor{SchoolColor}{HTML}{CC3333} % Crimson
%\definecolor{chaptergrey}{HTML}{CC3333} % for chapter numbers

%\definecolor{SchoolColor}{HTML}{D43333} % Crimson
%\definecolor{chaptergrey}{HTML}{D43333} % for chapter numbers

\definecolor{SchoolColor}{HTML}{CC0000} % Crimson
\definecolor{chaptergrey}{HTML}{CC0000} % for chapter numbers


\hypersetup{
    colorlinks,
    citecolor=SchoolColor,
    filecolor=black,
    linkcolor=black,
    urlcolor=SchoolColor,
}

% Formatting guidelines found in:
% http://www.gsas.harvard.edu/publications/form_of_the_phd_dissertation.php
\renewcommand{\frontmatter}{
	\input{frontmatter/personalize}
%frontmatter/titelblad
%	\dedicationpage
%	\maketitle
	\copyrightpage
    \contentspage
	% \listoffigures % optional
	\dedicationpage
	\acknowledgments
}

\renewcommand{\maketitle}{
    \pagenumbering{roman}
    \setcounter{page}{1}
	\thispagestyle{empty}
	\vspace*{\fill}
	\vspace{100pt}
	\begin{center}
	 \Huge \textcolor{SchoolColor}{Caught in action: interfacial and amyloid protein conformation revealed with vibrational spectroscopies} \normalsize \\
%	\Huge \textcolor{blue}{\thetitle} \normalsize \\
	\vspace{100pt}
%	\textsc{a dissertation presented \\ by\\
%	\theauthor\\ to\\ The \@department\\
%	\vspace{12pt}
%	in partial fulfillment of the requirements\\
%	for the degree of\\ \@degree\\
%	in the subject of\\ \@field\\
%	\vspace{12pt}
%	\@university\\ \@universitycity, \@universitystate\\
%	\@degreemonth\ \@degreeyear}
	\end{center} \vspace*{\fill}
}

%\usepackage{showhyphens}

\newcommand{\copyrightpage}{
	\newpage
	\thispagestyle{empty}
	\vspace*{\fill}
	\scshape \noindent \small \copyright \small 2018\hspace{3pt}-- \theauthor \\
	\noindent all rights reserved.
	\vspace*{\fill}
	\newpage
	\rm
}

\newcommand{\abstractpage}{
    \pdfbookmark{Summary}{Summary}
	\newpage
%	\pagenumbering{roman}
%	\setcounter{page}{3}
	\pagestyle{fancy}
%	\lhead{Thesis advisor: Professor \@advisor} \rhead{\@author}
	\renewcommand{\headrulewidth}{0.0pt}
	\vspace*{30pt}
	\begin{center}
    	\Large \textcolor{SchoolColor}{Caught in action: interfacial and amyloid protein conformation revealed with vibrational spectroscopies} \normalsize \\
    	\vspace*{10pt}
    	\scshape Summary \\ \rm
	\end{center}
%    \begin{spacing}{\dnormalspacing}
    	\input{endmatter/abstract}
%    \end{spacing}
	\vspace*{\fill}
	\newpage \lhead{} \rhead{}
	\cfoot{\thepage}
}


\newcommand{\samenvattingpage}{
	\pdfbookmark{Samenvatting}{Samenvatting}
	\newpage
	%	\pagenumbering{roman}
	%	\setcounter{page}{3}
	\pagestyle{fancy}
	%	\lhead{Thesis advisor: Professor \@advisor} \rhead{\@author}
	\renewcommand{\headrulewidth}{0.0pt}
	\vspace*{30pt}
	\begin{center}
		\Large \textcolor{SchoolColor}{Vastgelegd in actie: hoe de conformatie van eiwitten aan grensvlakken en in amylo\"{i}de structuren te bestuderen bestudeerd kan worden met vibrationele spectroscopie} \normalsize \\
		\vspace*{10pt}
		\scshape Samenvatting \\ \rm
	\end{center}
	%    \begin{spacing}{\dnormalspacing}
	\input{endmatter/samenvatting}
	%    \end{spacing}
	\vspace*{\fill}
	\newpage \lhead{} \rhead{}
	\cfoot{\thepage}
}


\newcommand{\contentspage}{
    \pdfbookmark{\contentsname}{Contents}
    \tableofcontents
}

\newcommand{\dedicationpage}[1]{
    \cleardoublepage
    \phantomsection
    \pdfbookmark{Dedication}{Dedication}
%	\newpage \thispagestyle{fancy} \vspace*{\fill}
	\newpage %\vspace*{\fill}
	\scshape \noindent \input{#1}
	%\vspace*{\fill} 
	\newpage \rm
    \cleardoublepage
}



\newcommand{\publicationspage}{
	%\cleardoublepage
	\phantomsection
	\begin{center}
		{\color{SchoolColor}\Titulosize\bfseries Unraveling the elusive solvation structure of aqueous ions using advanced spectroscopic techniques} \normalsize \\
		\vspace*{20pt}
		{\large\bfseries Publications covered in this thesis:} \\ \rm
	\end{center}
	    \begin{spacing}{\dnormalspacing}
		\input{frontmatter/publicationslistinthesis}
	\end{spacing}
	\begin{center}
	\vspace*{20pt}
	{\large\bfseries Other collaborations:} \\ \rm
\end{center}
\begin{spacing}{\dnormalspacing}
	\input{frontmatter/publicationslistnotinthesis}
\end{spacing}
	\vspace*{\fill}
	\newpage \lhead{} \rhead{}
	\cfoot{\thepage}
}

\newcommand{\acknowledgments}{
	\chapter*{Acknowledgments}
	\noindent
    \begin{spacing}{\dnormalspacing}
    	\input{frontmatter/thanks}
    \end{spacing}
	\vspace*{\fill} \newpage
	\setcounter{page}{1}
	\pagenumbering{arabic}
}

%Achtergrondkleur
%\usepackage[most]{tcolorbox}
%
%\tcbset{
%	frame code={}
%	center title,
%	left=0pt,
%	right=0pt,
%	top=0pt,
%	bottom=0pt,
%	colback=gray!70,
%	colframe=white,
%	width=\dimexpr\textwidth\relax,
%	enlarge left by=0mm,
%	boxsep=5pt,
%	arc=0pt,outer arc=0pt,
%}

%\RequirePackage{sectsty}
%\usepackage{fancyhdr}
%\pagestyle{fancy}
%\renewcommand{\chaptermark}[1]{\markboth{#1}{#1}}
%\renewcommand{\sectionmark}[1]{\markright{\thesection\ #1}}
%\lhead[\fancyplain{}{\textsf{\thepage}}]{\fancyplain{}{\textsf{\nouppercase\rightmark}}}
%\rhead[\fancyplain{}{\textsf{\nouppercase\leftmark}}]{\fancyplain{}{\textsf{\thepage}}}

% To use m in tabular:
\usepackage{array}
\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}

\usepackage{tabularx}
\usepackage{dcolumn}
% BibLaTeX
%\usepackage[sorting=none]{biblatex}

% jabbrv
\usepackage{packages/jabbrv}
%\usepackage[warnundef]{jabbrv} %[maxnames=5]

%natbib
%\usepackage[sort&compress,super,numbers]{natbib}

% Multibib
\usepackage{multibib}
\newcites{epilogue}{Epilogue References}
%\usepackage{cite}
% Back matter
\renewcommand{\backmatter}{
    \begin{appendices}
        \include{chapters/appendixA}
    \end{appendices}
    \input{endmatter/personalize}
    \clearpage
    \begin{spacing}{\dcompressedspacing}
%    	\small
%        \bibliography{references}
%\addbibresource{references} % Use when using biblatex
%        \addcontentsline{toc}{chapter}{References}
%        \bibliographystyle{apalike2}
%        \bibliographystyle{naturemag}
%        \bibliographystyle{abbrv}
%\bibliographystyle{jabbrv_abbrv}
%\bibliographystyle{jabbrv_unsrt}
%\bibliographystyle{jabbrv_acm}
%\bibliographystyle{jabbrv_alpha}
%\bibliographystyle{jabbrv_plain}
%\bibliographystyle{jabbrv_siam}
%\bibliographystyle{jabbrv_ieeetr}
%	    \include{endmatter/abstract}
%        \include{endmatter/colophon}
    \end{spacing}
}

%\newcommand{\abstractpage}{
%	\pdfbookmark{Abstract}{Abstract}
%	\newpage
%	\pagenumbering{roman}
%	\setcounter{page}{3}
%	\pagestyle{fancy}
%%	\lhead{Thesis advisor: Professor \@advisor} \rhead{\@author}
%	\renewcommand{\headrulewidth}{0.0pt}
%	\vspace*{35pt}
%	\begin{center}
%		\Large \textcolor{SchoolColor}{Caught in action: interfacial and amyloid protein conformation revealed with vibrational spectroscopy} \normalsize \\
%		\vspace*{20pt}
%		\scshape Abstract \\ \rm
%	\end{center}
%	\begin{spacing}{\dnormalspacing}
%		\input{frontmatter/abstract}
%	\end{spacing}
%	\vspace*{\fill}
%	\newpage \lhead{} \rhead{}
%	\cfoot{\thepage}
%}

	
% Plusminus sign:
\newcommand{\mypm}{\mathbin{\mathpalette\@mypm\relax}}
\newcommand{\@mypm}[2]{\ooalign{%
		\raisebox{.1\height}{$#1+$}\cr
		\smash{\raisebox{-.6\height}{$#1-$}}\cr}}

\newcommand{\onlinecite}[1]{\hspace{-1 ex} \nocite{#1}\citenum{#1}}
\newcommand{\citeepiloguenum}[1]{\hspace{-1 ex} \nocite{#1}\citenum{#1}}

%\newcommand{\onlineciteje}[1]{\hspace{-1 ex} \nocite{#1}\citenum{#1}} 

\setcounter{secnumdepth}{4}


% Stuff from Tibi:
%\documentclass[10.5pt]{book}

% for testing, only build a few chapters
%\includeonly{frontmatter/frontmatter}
%\includeonly{intro/intro}
%\includeonly{water_igepal/water_igepal}
%\includeonly{production/production}
%\pdfminorversion=5 
%\pdfcompresslevel=9
%\pdfobjcompresslevel=2
%\excludeonly{<filelist>}               % inverse of above
%\renewcommand{\includegraphics}[2][]{} % no figures (this should be after importing graphicx
%\usepackage[draft]{graphicx}
%\usepackage[english,dutch]{babel}    
%\usepackage{graphicx}                   % for including figures
%%\usepackage{amsmath}                    % equations
%\usepackage{amssymb}                    % for the \nrightarrow in chem eq
%%\usepackage{bm}                         % bold maths symbols (eg. vectors)
%%\usepackage{nicefrac}                   % for fancy fractions (xfrac is better)
%%\usepackage{paralist}                   % for inline lists
%\usepackage{tabularx}                   % fit columns to table width
%\usepackage{booktabs}                   % professional looking tables
%\usepackage{wasysym}                    % for the \diameter symbol in sketches
%\usepackage{newfloat}
%\usepackage{bm} %bold face in mathmode
%%\newcommand{\onlinecite}[1]{\hspace{-1 ex} \nocite{#1}\citenum{#1}} 
%\usepackage{sidecap}
%\sidecaptionvpos{figure}{t}
%\usepackage{float}
%% numberic columns centered on the decimal place
%\usepackage{dcolumn}
%\newcolumntype{d}[1]{D{.}{.}{#1}}
%\usepackage{bpchem}
%% sansmath gives \sansmath command and environment, unlike sfmath which changes all fonts
%\usepackage{sansmath}                   % mathmode with sans fonts
%\usepackage{upgreek}
%\usepackage{calligra}
%\usepackage{multirow}
%%\setcitestyle{super}
%
%% fix the labels on the figures so that they are sf too
%%fix the placeholder to be half width
%\usepackage[sf,SF,bf,tight,TABTOPCAP]{subfigure}
%\renewcommand{\thesubfigure}{\alph{subfigure}}  % remove the brackets around the label
%\subfiglabelskip=0pt                    % center the label properly
%\setcounter{lofdepth}{1}                % list subfig captions in list of figures
%
%\usepackage[
%tophrase=dash,
%repeatunits=false,
%trapambigrange=false,
%obeyall,                                % follow family and math mode
%obeyfamily,                             % follow sf or rm family (still does math though)
%]{siunitx}                              % proper display of SI units
%%\usepackage[version=3]{mhchem}          % easy writing chemical formulas/reactions
%\usepackage{paralist}
%\usepackage{enumitem}
%\usepackage{enumerate}
%
%\usepackage[version=3]{mhchem}
%
%
%
%\makeatletter
%\g@addto@macro\@floatboxreset\centering
%\makeatother
%
%
%
%
%
%
%
%% NEED TO FIGURE SOMETHING BETTER OUT, THIS SCREWS WITH intertext SPACING TOO
%%\let\jotorig\jot
%\setlength{\jot}{14pt}                  % make the spacing between multiline eqns a bit bigger
%
%
%\usepackage{mythesis}                   % use Loic's style file
%
%\usepackage{myfuns}                     % use my functions
%
%\usepackage{diffcolours}                % add colours for latexdiff markup
%
%%\geometry{showcrop,showframe}
%
%
%\usepackage[
%%dvipdfm,
%breaklinks=true, 
%linkbordercolor={0 0 1}, 
%pdfborder={0 0 0}, 
%bookmarksnumbered=true,
%]{hyperref}
%\hypersetup{
%	unicode=true,                       % for mhchem bookmarks
%	pdfauthor = {Steven J. Roeters},
%	pdftitle = {Catching proteins in action with vibrational spectroscopy},
%	pdfsubject = {},
%	pdfkeywords = {Vibrational spectroscopy, amyloids, membrane proteins, VSFG, IR, FTIR, 1D-IR, 2D-IR, VCD}
%}
%
%
%\usepackage{longtable}

% Footnote that doesn't show up in TOC:
\makeatletter
\def\myfnt{\ifx\protect\@typeset@protect\expandafter\footnote\else\expandafter\@gobble\fi}
\makeatother

\hyphenation{spectro-scopy}
\hyphenation{gene-ration}
\hyphenation{mono-mers}
\hyphenation{mono-meric}
\hyphenation{fre-quency}
\hyphenation{trun-cations}
\hyphenation{form-ing}
\hyphenation{grens-vlakken}
\hyphenation{analyse-methoden}
\hyphenation{hydro-fobines}
\hyphenation{op-per-vlak-te-ac-tieve}
\hyphenation{toe-komst}
\hyphenation{spec-trum}
\hyphenation{filamen-teuze}